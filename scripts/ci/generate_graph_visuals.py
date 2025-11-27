#!/usr/bin/env python3
"""Generate molecular graph visualisations for CI artifacts.

The script loads a subset of molecules via :class:`GraphDataset` and emits
paired 2‑D depictions (RDKit when available, otherwise a placeholder) and 3‑D
interactive views.  Whenever the optional BuildAMol dependency is available we
delegate the 3‑D scene construction to it to obtain a polished molecular viewer;
otherwise the script falls back to RDKit/py3Dmol-style HTML or Plotly scenes
constructed directly from the graph topology.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import struct
import zlib
from itertools import cycle, islice
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import math
import numbers
import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from utils import gym_compat
except Exception:  # pragma: no cover - compatibility helper is optional
    gym_compat = None

class _StubGraphData:
    def __init__(self, num_nodes: int):
        self._num_nodes = max(int(num_nodes), 0)
        self.edge_index: list[list[int]] = [[], []]
        self.pos = None
        self.x = None

    def num_nodes(self) -> int:
        return self._num_nodes


class _StubGraphDataset:
    def __init__(self, smiles: List[str]):
        self.smiles = smiles
        self.graphs = [_StubGraphData(max(len(s), 1)) for s in smiles]
        self.labels = None

    @staticmethod
    def _read_smiles(path: str, limit: Optional[int]) -> List[str]:
        smiles: List[str] = []
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "smiles" not in [name.lower() for name in reader.fieldnames]:
                raise ValueError("CSV must include a 'smiles' column")
            canonical = None
            for name in reader.fieldnames:
                if name.lower() == "smiles":
                    canonical = name
                    break
            for row in reader:
                value = row.get(canonical or "smiles")
                if not value:
                    continue
                smiles.append(value.strip())
                if limit is not None and len(smiles) >= limit:
                    break
        if not smiles:
            raise ValueError(f"No SMILES found in {path}")
        return smiles

    @classmethod
    def from_csv(cls, path: str, n_rows: Optional[int] = None) -> "_StubGraphDataset":
        return cls(cls._read_smiles(path, n_rows))

    @classmethod
    def from_directory(
        cls,
        directory: str,
        ext: str = "csv",
        max_graphs: Optional[int] = None,
    ) -> "_StubGraphDataset":
        if ext != "csv":
            raise RuntimeError("Fallback GraphDataset only supports CSV inputs")
        entries = sorted(entry for entry in os.listdir(directory) if entry.lower().endswith(".csv"))
        if not entries:
            raise FileNotFoundError(f"No CSV files found under {directory}")
        return cls.from_csv(os.path.join(directory, entries[0]), n_rows=max_graphs)

    @classmethod
    def from_parquet(cls, *_args, **_kwargs):  # pragma: no cover - unsupported in fallback
        raise RuntimeError("Parquet loading requires optional numpy/pandas dependencies")


try:
    from data.mdataset import GraphData as _RealGraphData, GraphDataset as _RealGraphDataset

    GRAPH_DATASET_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when optional deps missing
    _RealGraphDataset = None
    _RealGraphData = None
    GRAPH_DATASET_AVAILABLE = False

GraphData = _RealGraphData or _StubGraphData  # type: ignore[assignment]
GraphDataset = _RealGraphDataset or _StubGraphDataset  # type: ignore[assignment]
FALLBACK_GRAPH_DATASET = _StubGraphDataset
_SYNTHETIC_SMILES = [
    "C",
    "CC",
    "C1=CC=CC=C1",
    "CCO",
    "C#N",
    "CC(=O)O",
]

_FORCE_FALLBACK_LOADER = os.environ.get("GRAPH_VISUALS_FORCE_FALLBACK", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_RDKit_SPEC = importlib.util.find_spec("rdkit")
RDKit_INSTALLED = bool(_RDKit_SPEC)
RDKit_IMPORT_ERROR: Optional[str] = None

try:  # pragma: no cover - optional dependency in CI
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    RDKit_AVAILABLE = True
except Exception as exc:  # pragma: no cover - gracefully degrade in environments w/o RDKit
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    rdDepictor = None  # type: ignore
    rdMolDraw2D = None  # type: ignore
    RDKit_AVAILABLE = False
    RDKit_IMPORT_ERROR = str(exc)

if RDKit_AVAILABLE:  # pragma: no cover - depends on optional rdkit
    _DRAW_COLOR_CLASS = getattr(rdMolDraw2D, "DrawColour", None) or getattr(rdMolDraw2D, "Color", None)
else:  # pragma: no cover - rdkit unavailable
    _DRAW_COLOR_CLASS = None

if gym_compat is not None:  # pragma: no cover - helper absent when utils unavailable
    gym_compat.ensure_gymnasium_alias()

try:  # pragma: no cover - optional dependency in CI
    import buildamol

    BUILDAMOL_AVAILABLE = True
except Exception:  # pragma: no cover - degrade gracefully when BuildAmol is absent
    buildamol = None  # type: ignore[assignment]
    BUILDAMOL_AVAILABLE = False

logger = logging.getLogger(__name__)

_MATPLOTLIB_AVAILABLE = bool(importlib.util.find_spec("matplotlib"))


def _default_dataset_dir() -> Optional[str]:
    return os.environ.get("DATASET_DIR")


def _default_output_dir() -> Optional[str]:
    root = os.environ.get("PRETRAIN_EXPERIMENT_ROOT")
    if not root:
        return None
    return os.path.join(root, "graphs")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 2‑D and 3‑D molecular graph visuals for CI artifacts",
    )
    parser.add_argument(
        "--dataset-path",
        default=_default_dataset_dir(),
        help="Path to the dataset directory or file (defaults to DATASET_DIR env; synthetic fallback when unset)",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="Directory that will store generated .png/.html files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of molecules to visualise (default: 8)",
    )
    args = parser.parse_args(argv)
    if not args.output_dir:
        parser.error("--output-dir not provided and PRETRAIN_EXPERIMENT_ROOT is unset")
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be > 0")
    if not args.dataset_path:
        logger.warning("DATASET_DIR is unset; generating synthetic graph visuals")
    return args


def _guess_extension(path: str) -> str:
    if os.path.isfile(path):
        return os.path.splitext(path)[1].lstrip(".").lower()
    if not os.path.isdir(path):
        raise FileNotFoundError(f"dataset path {path} not found")
    entries = os.listdir(path)
    for candidate in ("parquet", "csv"):
        for entry in entries:
            if entry.lower().endswith(f".{candidate}"):
                return candidate
    raise ValueError(
        f"Could not infer dataset file extension under {path}; supported: csv, parquet"
    )


def _synthetic_dataset(limit: Optional[int]) -> Tuple[GraphDataset, str]:
    target = limit if (limit is not None and limit > 0) else len(_SYNTHETIC_SMILES)
    target = max(target, 1)
    smiles = list(islice(cycle(_SYNTHETIC_SMILES), target))
    logger.warning(
        "Falling back to %d synthetic molecules for graph visualisation", target
    )
    return FALLBACK_GRAPH_DATASET(smiles), "synthetic"


def _graph_count(dataset: Any) -> int:
    graphs = getattr(dataset, "graphs", None)
    if graphs is None:
        return 0
    try:
        return len(graphs)
    except Exception:
        pass
    try:
        return int(getattr(dataset, "num_graphs"))
    except Exception:
        return 0


def _load_dataset(dataset_path: Optional[str], limit: Optional[int]) -> Tuple[GraphDataset, str]:
    if not dataset_path:
        return _synthetic_dataset(limit)

    dataset_path = os.path.abspath(dataset_path)
    try:
        ext = _guess_extension(dataset_path)
    except Exception as exc:
        logger.error(
            "Unable to inspect dataset path %s (%s); using synthetic molecules",
            dataset_path,
            exc,
        )
        return _synthetic_dataset(limit)
    limit = None if (limit is None or limit <= 0) else limit
    loader_label = "fallback"
    use_fallback = _FORCE_FALLBACK_LOADER or not GRAPH_DATASET_AVAILABLE
    if not use_fallback:
        try:
            loader_label = "graphdataset"
            if os.path.isdir(dataset_path):
                logger.info("Loading dataset directory %s (ext=%s)", dataset_path, ext)
                dataset = GraphDataset.from_directory(
                    dataset_path,
                    ext=ext,
                    max_graphs=limit,
                )
            elif ext == "parquet":
                logger.info("Loading dataset file %s (parquet)", dataset_path)
                dataset = GraphDataset.from_parquet(dataset_path, n_rows=limit)
            elif ext == "csv":
                logger.info("Loading dataset file %s (csv)", dataset_path)
                dataset = GraphDataset.from_csv(dataset_path, n_rows=limit)
            else:
                raise ValueError(f"Unsupported dataset extension: {ext}")

            count = _graph_count(dataset)
            if count <= 0:
                logger.warning(
                    "GraphDataset loader produced zero graphs for %s; falling back to CSV parser",
                    dataset_path,
                )
                use_fallback = True
            else:
                return dataset, loader_label
        except Exception as exc:
            logger.warning(
                "GraphDataset loader failed for %s (%s); falling back to CSV-only parser",
                dataset_path,
                exc,
            )
            use_fallback = True
    if use_fallback:
        try:
            if os.path.isdir(dataset_path):
                logger.info(
                    "Loading dataset directory %s via fallback loader", dataset_path
                )
                dataset = FALLBACK_GRAPH_DATASET.from_directory(
                    dataset_path,
                    ext=ext,
                    max_graphs=limit,
                )
            elif ext == "csv":
                logger.info(
                    "Loading dataset file %s via fallback loader", dataset_path
                )
                dataset = FALLBACK_GRAPH_DATASET.from_csv(dataset_path, n_rows=limit)
            else:
                raise ValueError(
                    "Fallback graph loader only supports CSV inputs; set DATASET_DIR to a CSV file"
                )
            return dataset, "fallback"
        except Exception as exc:
            logger.error(
                "Fallback graph loader failed for %s (%s); using synthetic molecules",
                dataset_path,
                exc,
            )
            return _synthetic_dataset(limit)
    raise RuntimeError("Unexpected dataset loading state")


def _select_indices(total: int, limit: int) -> List[int]:
    if total <= 0 or limit <= 0:
        return []
    if limit >= total:
        return list(range(total))
    if limit == 1:
        return [0]
    step = (total - 1) / float(limit - 1)
    indices = set()
    for i in range(limit):
        candidate = int(round(step * i))
        candidate = max(0, min(total - 1, candidate))
        indices.add(candidate)
    return sorted(indices)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coerce_label(labels: Optional[Sequence[Any]], idx: int) -> Optional[float]:
    if labels is None:
        return None
    length: Optional[int] = None
    try:
        length = len(labels)
    except Exception:
        pass
    if length is not None and idx >= length:
        return None
    try:
        value = labels[idx]
    except Exception:
        return None
    if isinstance(value, numbers.Number):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _render_placeholder_png(path: Path, caption: str) -> None:
    _ensure_dir(path.parent)

    width, height = 320, 240
    background = (245, 245, 245, 255)
    border = (190, 190, 190, 255)
    accent = (207, 34, 46, 255)

    pixels = bytearray()
    for y in range(height):
        pixels.append(0)  # per-row filter type "None"
        for x in range(width):
            color = background
            if x in {0, width - 1} or y in {0, height - 1}:
                color = border
            elif abs(x - width // 2) <= 2 or abs(y - height // 2) <= 2:
                color = accent
            pixels.extend(color)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    compressed = zlib.compress(bytes(pixels))
    png_bytes = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", compressed) + _chunk(
        b"IEND", b""
    )
    path.write_bytes(png_bytes)

    caption_path = path.with_suffix(".txt")
    caption_path.write_text(caption.rstrip() + "\n", encoding="utf-8")


def _prepare_highlight_colors(mol: "Chem.Mol"):
    ring_info = Chem.GetSymmSSSR(mol)
    highlight_atoms: set[int] = set()
    highlight_bonds: set[int] = set()
    for ring in ring_info:
        atoms = [int(a) for a in ring]
        highlight_atoms.update(atoms)
        for start, end in zip(atoms, atoms[1:] + atoms[:1]):
            bond = mol.GetBondBetweenAtoms(start, end)
            if bond is not None:
                highlight_bonds.add(bond.GetIdx())
    if not _DRAW_COLOR_CLASS:
        return highlight_atoms, highlight_bonds, {}, {}
    atom_colors = {idx: _DRAW_COLOR_CLASS(0.99, 0.42, 0.09) for idx in highlight_atoms}
    bond_colors = {idx: _DRAW_COLOR_CLASS(0.13, 0.57, 0.27) for idx in highlight_bonds}
    return highlight_atoms, highlight_bonds, atom_colors, bond_colors


def _draw_matplotlib_graph(graph: GraphData, smiles: Optional[str], path: Path) -> bool:
    if not _MATPLOTLIB_AVAILABLE:
        return False
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # type: ignore

    positions = _graph_positions(graph)
    if not positions:
        return False
    edges = _unique_edges(graph)
    try:
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")
        xs, ys, zs = zip(*positions)
        ax.scatter(xs, ys, zs, c="#0d47a1", alpha=0.85, s=40, depthshade=True)
        for src, dst in edges:
            if src >= len(positions) or dst >= len(positions):
                continue
            ax.plot(
                [positions[src][0], positions[dst][0]],
                [positions[src][1], positions[dst][1]],
                [positions[src][2], positions[dst][2]],
                color="#9e9e9e",
                linewidth=1.5,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if smiles:
            ax.set_title(smiles)
        fig.tight_layout()
        _ensure_dir(path.parent)
        fig.savefig(path, dpi=150)
    except Exception:
        return False
    finally:
        plt.close("all")
    return True


def _draw_rdkit_2d(smiles: str, path: Path) -> bool:
    if not RDKit_AVAILABLE:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    rdDepictor.SetPreferCoordGen(True)  # type: ignore[arg-type]
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(480, 360)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.highlightRadius = 0.35
    opts.dotsPerAngstrom = 45
    opts.bondLineWidth = 2.2
    opts.useBWAtomPalette()  # start from greyscale so highlights stand out
    highlight_atoms, highlight_bonds, atom_colors, bond_colors = _prepare_highlight_colors(mol)
    try:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_atoms),
            highlightBonds=list(highlight_bonds),
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
    except Exception:
        return False
    _ensure_dir(path.parent)
    with open(path, "wb") as handle:
        handle.write(png)
    return True


def _to_sequence(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            coerced = value.tolist()
            if isinstance(coerced, list):
                return coerced
        except Exception:
            pass
    try:
        return list(value)
    except Exception:
        return None


def _normalise_matrix(value: Any) -> Optional[List[List[float]]]:
    rows = _to_sequence(value)
    if rows is None:
        return None
    matrix: List[List[float]] = []
    for row in rows:
        cols = _to_sequence(row)
        if cols is None:
            cols = [row]
        parsed: List[float] = []
        for col in cols:
            try:
                parsed.append(float(col))
            except Exception:
                parsed.append(0.0)
        if parsed:
            matrix.append(parsed)
    if not matrix:
        return None
    return matrix


def _pad_matrix(matrix: List[List[float]], width: int) -> List[List[float]]:
    padded: List[List[float]] = []
    for row in matrix:
        current = list(row[:width])
        if len(current) < width:
            current.extend([0.0] * (width - len(current)))
        padded.append(current)
    return padded


def _graph_positions(graph: GraphData) -> List[List[float]]:
    for attr in ("pos", "x"):
        candidate = _normalise_matrix(getattr(graph, attr, None))
        if candidate:
            width = len(candidate[0])
            if width >= 1:
                return _pad_matrix(candidate, 3)
    n = max(int(graph.num_nodes()), 0)
    if n <= 0:
        return []
    theta_step = 2 * math.pi / n
    positions: List[List[float]] = []
    for idx in range(n):
        theta = theta_step * idx
        x_coord = math.cos(theta)
        y_coord = math.sin(theta)
        z_coord = -0.5 + (idx / max(1, n - 1))
        positions.append([x_coord, y_coord, z_coord])
    return positions


def _unique_edges(graph: GraphData) -> List[tuple[int, int]]:
    edge_index = getattr(graph, "edge_index", None)
    rows = _normalise_matrix(edge_index)
    if not rows or len(rows) < 2:
        return []
    src_row, dst_row = rows[0], rows[1]
    length = min(len(src_row), len(dst_row))
    pairs = set()
    for i in range(length):
        try:
            src = int(src_row[i])
            dst = int(dst_row[i])
        except Exception:
            continue
        if src == dst:
            continue
        key = tuple(sorted((src, dst)))
        pairs.add(key)
    return sorted(pairs)


def _plotly_html(graph: GraphData, smiles: Optional[str], div_id: str) -> str:
    positions = _graph_positions(graph)
    nodes = positions
    tooltips = []
    for idx, coords in enumerate(nodes):
        tooltip = f"Atom {idx}"
        if smiles:
            tooltip += f" | {smiles}"
        tooltips.append(tooltip)
    node_trace = {
        "type": "scatter3d",
        "mode": "markers",
        "x": [p[0] for p in nodes],
        "y": [p[1] for p in nodes],
        "z": [p[2] for p in nodes],
        "text": tooltips,
        "marker": {
            "size": 7,
            "color": "#0d47a1",
            "opacity": 0.85,
        },
        "name": "atoms",
    }
    edges = _unique_edges(graph)
    edge_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "x": [],
        "y": [],
        "z": [],
        "line": {"width": 2, "color": "#9e9e9e"},
        "name": "bonds",
    }
    for src, dst in edges:
        if src >= len(nodes) or dst >= len(nodes):
            continue
        edge_trace["x"].extend([nodes[src][0], nodes[dst][0], None])
        edge_trace["y"].extend([nodes[src][1], nodes[dst][1], None])
        edge_trace["z"].extend([nodes[src][2], nodes[dst][2], None])
    layout = {
        "title": f"Molecular graph {smiles or ''}".strip(),
        "scene": {
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "z"},
        },
        "showlegend": False,
    }
    import json as _json

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'/>
<title>Graph visualisation</title>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
</head>
<body>
<div id='{div_id}' style='width: 640px; height: 480px;'></div>
<script>
const data = {_json.dumps([edge_trace, node_trace])};
const layout = {_json.dumps(layout)};
Plotly.newPlot('{div_id}', data, layout, {{displaylogo: false}});
</script>
</body>
</html>"""


def _build_buildamol_3d(smiles: str) -> Optional[str]:
    if not BUILDAMOL_AVAILABLE:
        return None
    try:
        molecule = buildamol.molecule(smiles)
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        logger.debug("BuildAmol molecule() failed for %s: %s", smiles, exc)
        return None
    try:
        viewer = molecule.plotly()
        figure = getattr(viewer, "figure", None)
        if figure is None:
            return None
        return figure.to_html(include_plotlyjs="cdn", full_html=True)
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        logger.debug("BuildAmol viewer generation failed for %s: %s", smiles, exc)
        return None


def _build_rdkit_3d(smiles: str) -> Optional[str]:
    if not RDKit_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        return None
    try:
        block = Chem.MolToMolBlock(mol)
    except Exception:
        return None
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'/>
<title>{smiles}</title>
<script src='https://3dmol.org/build/3Dmol-min.js'></script>
</head>
<body>
<div id='viewer' style='width:640px;height:480px;position:relative;'></div>
<script>
var viewer = $3Dmol.createViewer('viewer', {{ backgroundColor: 'white' }});
viewer.addModel(`{block}`, 'sdf');
viewer.setStyle({{}}, {{stick: {{colorscheme: 'cyanCarbon'}}}});
viewer.zoomTo();
viewer.render();
</script>
</body>
</html>"""


def _write_html(path: Path, html: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(html, encoding="utf-8")


def _render_sample(
    record_dir: Path,
    graph: GraphData,
    smiles: Optional[str],
    label: Optional[float],
    index: int,
) -> str:
    png_path = record_dir / "molecule.png"
    html_path = record_dir / "molecule.html"
    rendered_png = False
    renderer = "placeholder"
    if smiles and _draw_rdkit_2d(smiles, png_path):
        logger.info("Rendered RDKit 2‑D visual for %s", smiles)
        rendered_png = True
        renderer = "rdkit"
    elif _draw_matplotlib_graph(graph, smiles, png_path):
        logger.info("Rendered matplotlib fallback for sample %s", smiles or index)
        rendered_png = True
        renderer = "matplotlib"
    if not rendered_png:
        if not smiles:
            caption = f"No SMILES provided for sample {index}"
        elif not RDKit_AVAILABLE:
            if RDKit_INSTALLED and RDKit_IMPORT_ERROR:
                caption = "RDKit import failed; placeholder molecule"
            else:
                caption = "RDKit unavailable; placeholder molecule"
        else:
            caption = f"RDKit rendering failed for sample {index}"
        _render_placeholder_png(png_path, caption=caption)
        if not RDKit_AVAILABLE:
            if RDKit_INSTALLED and RDKit_IMPORT_ERROR:
                logger.info(
                    "RDKit import failed (%s); wrote placeholder PNG for sample %s",
                    RDKit_IMPORT_ERROR,
                    smiles or index,
                )
            else:
                logger.info(
                    "RDKit unavailable; wrote placeholder PNG for sample %s", smiles or index
                )
        else:
            logger.info("RDKit rendering failed; wrote placeholder PNG for sample %s", smiles or index)
    html_payload = None
    if smiles:
        html_payload = _build_buildamol_3d(smiles)
        if not html_payload:
            html_payload = _build_rdkit_3d(smiles)
    if not html_payload:
        html_payload = _plotly_html(graph, smiles, div_id=f"plot_{index:03d}")
    _write_html(html_path, html_payload)
    metadata = {
        "index": index,
        "smiles": smiles,
        "num_nodes": graph.num_nodes(),
        "num_edges": len(_unique_edges(graph)),
        "label": label,
    }
    (record_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return renderer


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[graph-visuals] %(message)s")
    args = parse_args(argv)
    dataset, loader_label = _load_dataset(args.dataset_path, args.num_samples)
    dataset_label = args.dataset_path or "synthetic"
    total = len(dataset.graphs)
    if total == 0:
        logger.warning("Dataset %s contained zero graphs; nothing to render", dataset_label)
        return 0
    limit = min(args.num_samples, total)
    indices = _select_indices(total, limit)
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)
    logger.info("Generating graph visuals for %d / %d molecules", len(indices), total)
    png_stats = {"rdkit": 0, "matplotlib": 0, "placeholder": 0}
    rdkit_placeholder_count = 0
    for slot, dataset_idx in enumerate(indices):
        record_dir = output_dir / f"sample_{slot:03d}"
        graph = dataset.graphs[dataset_idx]
        smiles = None
        if dataset.smiles and dataset_idx < len(dataset.smiles):
            smiles = dataset.smiles[dataset_idx]
        label = _coerce_label(dataset.labels, dataset_idx)
        renderer = _render_sample(record_dir, graph, smiles, label, dataset_idx)
        png_stats[renderer] = png_stats.get(renderer, 0) + 1
        if renderer == "placeholder" and not RDKit_AVAILABLE:
            rdkit_placeholder_count += 1
    if not RDKit_AVAILABLE:
        if rdkit_placeholder_count:
            logger.warning(
                "RDKit unavailable; generated %d placeholder PNG(s). Install rdkit to render 2-D depictions.",
                rdkit_placeholder_count,
            )
        else:
            logger.info(
                "RDKit unavailable; matplotlib fallback rendered all samples."
            )
    summary = {
        "dataset_path": os.path.abspath(dataset_label)
        if args.dataset_path
        else "synthetic",
        "output_dir": str(output_dir.resolve()),
        "num_graphs": total,
        "num_rendered": len(indices),
        "loader": loader_label,
        "fallback_forced": bool(_FORCE_FALLBACK_LOADER),
        "rdkit_available": bool(RDKit_AVAILABLE),
        "rdkit_installed": bool(RDKit_INSTALLED),
        "rdkit_import_error": RDKit_IMPORT_ERROR,
        "png_renderers": png_stats,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Graph visuals ready under %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

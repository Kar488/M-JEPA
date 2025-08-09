from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict

import yaml
import logging

logger = logging.getLogger(__name__)

class BaselineCLI:
    def __init__(self, cfg_path: str = "adapters/config.yaml"):
        with open(cfg_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        self.paths: Dict[str, str] = y["paths"]
        self.outputs_root: str = y["outputs"]["root"]
        self.cmds: Dict[str, Dict[str, str]] = y["commands"]

    def _format(self, method: str, template: str, **kwargs) -> str:
        return template.format(repo=self.paths[method], **kwargs)

    def train(self, method: str, unlabeled: str, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cmd = self._format(
            method, self.cmds[method]["train"], unlabeled=unlabeled, out=out_dir
        )
        logger.info("[%s] TRAIN:\n  %s", method, cmd)
        subprocess.run(shlex.split(cmd), check=True)

    def embed(
        self, method: str, ckpt_path: str, smiles_file: str, emb_out: str
    ) -> None:
        Path(emb_out).parent.mkdir(parents=True, exist_ok=True)
        cmd = self._format(
            method,
            self.cmds[method]["embed"],
            ckpt=ckpt_path,
            smiles=smiles_file,
            emb=emb_out,
        )
        logger.info("[%s] EMBED:\n  %s", method, cmd)
        subprocess.run(shlex.split(cmd), check=True)

    def outputs_dir(self, method: str) -> str:
        return str(Path(self.outputs_root) / method)

import math

import pytest

pytest.importorskip("numpy")

from experiments import grid_search as gs

np = pytest.importorskip("numpy")


def test_run_one_config_method_handles_empty_eval_dataset(tmp_path):
    cfg = gs.Config(
        mask_ratio=0.1,
        contiguous=False,
        hidden_dim=16,
        num_layers=2,
        gnn_type="gcn",
        ema_decay=0.99,
        add_3d=False,
        augmentations=gs.AugmentationConfig(),
        pretrain_bs=1,
        finetune_bs=1,
        pretrain_epochs=1,
        finetune_epochs=1,
        lr=1e-3,
        temperature=0.1,
    )

    unlabeled = gs._GraphDatasetShim(graphs=[object()])
    empty_eval = gs._GraphDatasetShim(graphs=[], labels=np.array([]))

    row = gs._run_one_config_method(
        cfg=cfg,
        method="jepa",
        unlabeled_dataset_fn=None,
        eval_dataset_fn=None,
        task_type="regression",
        seeds=[0],
        device="cpu",
        use_wandb=False,
        ckpt_dir=str(tmp_path),
        ckpt_every=1,
        use_scheduler=False,
        warmup_steps=0,
        baseline_unlabeled_file=None,
        baseline_eval_file=None,
        baseline_smiles_col="smiles",
        baseline_label_col=None,
        prebuilt_loaders=None,
        prebuilt_datasets=(unlabeled, empty_eval, empty_eval),
    )

    assert row["skip_reason"] == "empty_eval_dataset"
    assert math.isnan(row["rmse"]) and math.isnan(row["val_rmse"])
    assert "rmse" in row and "val_rmse" in row

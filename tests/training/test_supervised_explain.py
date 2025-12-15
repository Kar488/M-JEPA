from training import supervised


def test_normalise_explain_modes_handles_multiple_inputs():
    modes = supervised._normalise_explain_modes(" ig , motif_ig  ig")
    assert modes == ["ig", "ig_motif"]

    modes_from_iterable = supervised._normalise_explain_modes(["ig", "motif_ig", "ig"])
    assert modes_from_iterable == ["ig", "ig_motif"]


def test_build_explain_loggers_includes_all_requested(monkeypatch):
    calls = []

    class DummyIGLogger:
        def __init__(
            self,
            *,
            dataset,
            encoder,
            head_module,
            task_type,
            device,
            explain_mode,
            explain_config,
            stage_config,
        ) -> None:
            self.enabled = True
            calls.append(("ig_init", explain_mode, dataset, encoder, head_module, task_type, device))

        def process_batch(self, batch_meta):
            calls.append(("ig_batch", batch_meta))

        def finalize(self, metrics):
            metrics["ig_done"] = True

    class DummyMotifLogger:
        def __init__(
            self,
            *,
            dataset,
            encoder,
            head_module,
            task_type,
            device,
            explain_mode,
            explain_config,
            stage_config,
        ) -> None:
            self.enabled = True
            calls.append(("motif_init", explain_mode, dataset, encoder, head_module, task_type, device))

        def process_batch(self, batch_meta):
            calls.append(("motif_batch", batch_meta))

        def finalize(self, metrics):
            metrics["motif_done"] = True

    monkeypatch.setattr(supervised, "_IGArtifactLogger", DummyIGLogger)
    monkeypatch.setattr(supervised, "_MotifIGArtifactLogger", DummyMotifLogger)

    dataset = object()
    encoder = object()
    head = object()
    modes = ["ig", "ig_motif"]

    loggers = supervised._build_explain_loggers(
        modes=modes,
        dataset=dataset,
        encoder=encoder,
        head_module=head,
        task_type="classification",
        device=object(),
        explain_config={},
        stage_config={},
    )

    assert len(loggers) == 2
    metrics: dict = {}
    for logger in loggers:
        logger.process_batch({"batch_indices": [0]})
    for logger in loggers:
        logger.finalize(metrics)

    assert metrics == {"ig_done": True, "motif_done": True}
    assert ("ig_batch", {"batch_indices": [0]}) in calls
    assert ("motif_batch", {"batch_indices": [0]}) in calls

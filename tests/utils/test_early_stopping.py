from utils.early_stopping import EarlyStopping


def test_early_stopping_improvement():
    es = EarlyStopping(patience=3)
    values = [5.0, 4.0, 3.0, 2.0]
    for v in values:
        assert es.step(v) is False


def test_early_stopping_non_improvement():
    es = EarlyStopping(patience=3)
    values = [1.0, 1.0, 1.0, 1.0]
    results = [es.step(v) for v in values]
    assert results[:-1] == [False, False, False]
    assert results[-1] is True
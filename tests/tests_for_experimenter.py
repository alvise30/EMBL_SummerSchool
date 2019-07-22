import summer


def test_run_exists():
    assert hasattr(summer, "run")
    assert callable(summer.run)

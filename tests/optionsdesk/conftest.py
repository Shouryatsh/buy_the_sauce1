import pytest


@pytest.fixture(autouse=True)
def _ignore_project_config(monkeypatch, tmp_path):
    """Tests must not depend on the user's config.yaml (it is set to live)."""
    monkeypatch.setenv("OPTIONSDESK_CONFIG", str(tmp_path / "none.yaml"))

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_home():
    app = AppTest.from_file(Path("app.py").resolve()).run()
    assert not app.exception
    assert app.title[0].value == "RoadFusion AI"


def test_all_pages_without_assets(tmp_path,monkeypatch):
    monkeypatch.setenv("ROADFUSION_WEIGHTS",str(tmp_path/"missing.pt"))
    app=AppTest.from_file(Path("app.py").resolve(),default_timeout=15).run()
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception,page

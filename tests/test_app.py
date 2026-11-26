from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_home():
    app = AppTest.from_file(Path("app.py").resolve()).run()
    assert not app.exception
    assert app.title[0].value == "RoadFusion AI"


def test_all_pages_without_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADFUSION_WEIGHTS", str(tmp_path / "missing.pt"))
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=15).run()
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_edge_controls_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADFUSION_WEIGHTS", str(tmp_path / "missing.pt"))
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=15).run()
    app.sidebar.radio[0].set_value("Virtual Edge Device").run()

    def click(label):
        next(b for b in app.button if b.label == label).click().run()
        assert not app.exception

    click("Ramp step")
    assert app.session_state.edge_sim.rows[-1]["seconds"] is None
    click("Start ramp")
    click("Pause ramp")
    n = len(app.session_state.edge_sim.rows)
    app.run()
    assert len(app.session_state.edge_sim.rows) == n
    click("Reset simulation")
    assert not app.session_state.edge_sim.rows


def test_benchmark_panel_missing_and_corrupt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = "from core.benchmark_panel import panel\npanel()"
    app = AppTest.from_string(source).run()
    assert app.info and not app.exception
    report = Path("runs/environment_benchmarks/example/summary_metrics.json")
    report.parent.mkdir(parents=True)
    report.write_text("{")
    app.run()
    assert (
        "Cannot read environment benchmark" in app.error[0].value and not app.exception
    )

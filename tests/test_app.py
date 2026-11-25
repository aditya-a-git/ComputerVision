from streamlit.testing.v1 import AppTest


def test_home():
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert app.title[0].value == "RoadFusion AI"

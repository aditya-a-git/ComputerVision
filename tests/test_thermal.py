from core.thermal_controller import Controller


def test_heat_and_recovery():
    c = Controller()
    expected = [
        "full_fusion",
        "reduced_compute",
        "reduced_compute",
        "single_modality",
        "single_modality",
        "reduced_compute",
        "reduced_compute",
        "full_fusion",
    ]
    for t, mode in zip([50, 71, 86, 86, 82, 79, 67, 64], expected):
        assert c.step(t, (1, 1))[0] == mode
    assert c.step(40, (0, 0))[0] == "unavailable"
    assert c.step(40, (1, 0))[0] == "single_modality"

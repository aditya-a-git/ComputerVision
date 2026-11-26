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


def test_ramp_reset_and_routing(monkeypatch):
    import numpy as np
    from core import fusion
    from core.edge_lab import Simulation, aggregate

    sim = Simulation(playing=True)
    temps = [sim.advance(65, 90, 10) for _ in range(4)]
    assert temps == [65, 75, 85, 90] and not sim.playing
    observed = []
    monkeypatch.setattr(
        fusion,
        "predict",
        lambda model, img, size, conf, source: (
            observed.append((source, size)) or [],
            0.01,
        ),
    )
    image = np.full((24, 24, 3), 127, np.uint8)
    for temp, health, count, size in [
        (65, (1, 1), 2, 640),
        (75, (1, 1), 2, 320),
        (85, (1, 1), 2, 320),
        (90, (1, 1), 1, 320),
        (65, (0, 0), 0, 320),
        (65, (1, 0), 1, 320),
        (64, (1, 1), 2, 640),
    ]:
        observed.clear()
        sim.observe(temp, health, object(), image, 0.25, "test")
        assert len(observed) == count and all(call[1] == size for call in observed)
    assert (
        len(aggregate(sim.rows)) == 3
        and sum(r["samples"] for r in aggregate(sim.rows)) == 6
    )
    assert (
        aggregate([dict(route="full", seconds=t) for t in (0.1, 0.3)])[0][
            "processed_fps"
        ]
        == 5
    )
    sim = Simulation()
    assert not sim.rows and sim.advance(65, 90, 10) == 65


def test_threshold_boundaries_and_oscillation():
    c = Controller()
    for t, expected in [
        (69, "full_fusion"),
        (70, "reduced_compute"),
        (69, "reduced_compute"),
        (70, "reduced_compute"),
        (66, "reduced_compute"),
        (65, "full_fusion"),
        (85, "full_fusion"),
        (85, "single_modality"),
        (84, "single_modality"),
        (81, "single_modality"),
        (80, "reduced_compute"),
    ]:
        assert c.step(t, (1, 1))[0] == expected

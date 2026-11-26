"""Step-driven Streamlit virtual controller with a bounded temperature ramp."""

from dataclasses import dataclass, field
from statistics import mean

import pandas as pd
import streamlit as st

from core.fusion import execute
from core.metrics import latency
from core.thermal_controller import Controller


@dataclass
class Simulation:
    controller: Controller = field(default_factory=Controller)
    rows: list = field(default_factory=list)
    playing: bool = False
    temperature: int | None = None

    def advance(self, start, maximum, increment):
        if not 20 <= start <= maximum <= 100 or increment < 1:
            raise ValueError("Invalid ramp range or increment")
        self.temperature = (
            start
            if self.temperature is None
            else min(maximum, self.temperature + increment)
        )
        if self.temperature >= maximum:
            self.playing = False
        return self.temperature

    def observe(self, temp, health, model, image, confidence, identity):
        route, mode, size = self.controller.step(temp, health)
        previous = self.rows[-1] if self.rows else {}
        run, boxes = {}, []
        if model is not None and image is not None:
            boxes, run = execute(model, image, mode, health, size, confidence)
        branches = run.get("branches", [])
        reason = self.controller.reason
        if previous.get("health") and health != previous["health"]:
            reason += " · stream health changed"
        if run and not branches and mode != "unavailable":
            reason += " · no view has positive reliability"
        row = dict(
            step=previous.get("step", 0) + 1,
            temperature=temp,
            previous=previous.get("route"),
            route=route,
            reason=reason,
            health=health,
            branches=branches,
            count=len(branches),
            actual_mode=run.get("mode"),
            size=size,
            model=identity,
            device="cpu",
            seconds=run["seconds"] if branches else None,
            branch_seconds=run.get("branch_seconds", {}),
        )
        self.rows = (self.rows + [row])[-100:]
        return boxes, run


def aggregate(rows):
    groups = {}
    for row in rows:
        if row["seconds"] is not None:
            groups.setdefault(row["route"], []).append(row["seconds"])
    return [
        dict(
            route=route,
            samples=len(times),
            mean_ms=mean(times) * 1000,
            **latency(times),
        )
        for route, times in groups.items()
    ]


def panel(model, image, conf, identity, show_run):
    state = st.session_state
    if "edge_sim" not in state:
        state.edge_sim = Simulation()
    sim = state.edge_sim
    health = (
        st.slider("RGB health", 0.0, 1.0, 1.0),
        st.slider("Proxy health", 0.0, 1.0, 1.0),
    )
    manual = st.slider("Simulated device temperature (°C)", 20, 100, 50)
    start = st.number_input("Ramp start (°C)", 20, 100, 50)
    maximum = st.number_input("Ramp maximum (°C)", start, 100, max(start, 90))
    increment = st.number_input("Ramp increment (°C)", 1, 30, 5)
    controls = st.columns(5)
    manual_step = controls[0].button("Process frame", disabled=sim.playing)
    ramp_step = controls[1].button(
        "Ramp step", disabled=sim.playing or sim.temperature == maximum
    )

    def set_playing(value):
        sim.playing = value

    def reset():
        state.edge_sim = Simulation()
        state.pop("edge_result", None)

    controls[2].button(
        "Start ramp",
        disabled=sim.playing or sim.temperature == maximum,
        on_click=set_playing,
        args=(True,),
    )
    controls[3].button(
        "Pause ramp", disabled=not sim.playing, on_click=set_playing, args=(False,)
    )
    controls[4].button("Reset simulation", on_click=reset)

    def observe(temp):
        state.edge_result = (
            *sim.observe(temp, health, model, image, conf, identity),
            image,
        )

    if manual_step or ramp_step:
        temp = manual if manual_step else sim.advance(start, maximum, increment)
        observe(temp)

    @st.fragment(run_every=1.0 if sim.playing else None)
    def timeline():
        if sim.playing:
            observe(sim.advance(start, maximum, increment))
            if not sim.playing:
                st.rerun()
        if not sim.rows:
            st.info("Process a frame or start the ramp to record a controller step.")
            return
        current = sim.rows[-1]
        st.progress(
            (current["temperature"] - 20) / 80,
            text=f"Simulated temperature · {current['temperature']} °C",
        )
        st.write(
            f"Controller · **{current['route']}** · {current['size']} px · executed branches: {', '.join(current['branches']) or 'none'}"
        )
        st.caption(current["reason"])
        boxes, run, processed = state.edge_result
        if current["seconds"] is None:
            st.info("Detection performance: Not measured")
        elif run:
            show_run(processed, boxes, run, current["route"])
        history = pd.DataFrame(sim.rows).set_index("step")
        temperatures = history[["temperature"]].assign(
            reduced=sim.controller.reduced, single=sim.controller.single
        )
        st.line_chart(temperatures, y_label="Simulated °C")
        st.scatter_chart(history.reset_index(), x="step", y="route")
        st.dataframe(history)
        stats = aggregate(sim.rows)
        if stats:
            table = pd.DataFrame(stats).set_index("route")
            st.dataframe(table)
            st.bar_chart(table[["p50_ms", "p95_ms"]], y_label="Measured latency (ms)")
            st.bar_chart(table[["processed_fps"]], y_label="Processed frames/s")
        st.caption(
            "Last 100 steps. Small sample percentiles are descriptive. CPU timings include view preparation through final filtering; temperature is simulated."
        )

    timeline()

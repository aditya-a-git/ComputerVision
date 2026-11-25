"""Illustrative software temperature policy. Dwell is measured in processed steps."""

from dataclasses import dataclass


@dataclass
class Controller:
    reduced: float = 70
    single: float = 85
    hysteresis: float = 5
    dwell: int = 2
    level: int = 0
    age: int = 2

    def __post_init__(self):
        if not 0 < self.hysteresis < self.single - self.reduced or self.dwell < 1:
            raise ValueError("Invalid temperature thresholds or dwell")

    def step(self, temp, health):
        self.age += 1
        target = self.level
        if temp >= self.single:
            target = 2
        elif self.level == 2 and temp > self.single - self.hysteresis:
            target = 2
        elif temp >= self.reduced:
            target = 1
        elif self.level and temp > self.reduced - self.hysteresis:
            target = 1
        else:
            target = 0
        if target != self.level and self.age >= self.dwell:
            self.level, self.age = target, 0
        valid = sum(h > 0 for h in health)
        if valid == 0:
            return "unavailable", "unavailable", 320
        if valid == 1 or self.level == 2:
            return "single_modality", "best", 320
        if self.level == 1:
            return "reduced_compute", "late_fusion", 320
        return "full_fusion", "late_fusion", 640

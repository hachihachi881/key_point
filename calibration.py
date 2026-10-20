"""Personal calibration helpers for posture assessment."""

from dataclasses import dataclass
import math
import statistics


class CalibrationError(ValueError):
    """Raised when a reliable calibration baseline cannot be created."""


@dataclass(frozen=True)
class CalibrationBaseline:
    head_forward_ratio: float
    trunk_angle: float
    valid_frames: int
    total_frames: int
    head_mad: float
    trunk_mad: float

    @property
    def valid_ratio(self):
        if self.total_frames <= 0:
            return 0.0
        return self.valid_frames / self.total_frames


class CalibrationCollector:
    def __init__(self):
        self.head_values = []
        self.trunk_values = []
        self.total_frames = 0

    def add(self, assessment):
        """Add one frame when both posture indicators are available."""
        self.total_frames += 1
        head = assessment.get("head_forward_ratio")
        trunk = assessment.get("trunk_angle")
        if not (_is_finite(head) and _is_finite(trunk)):
            return False

        self.head_values.append(float(head))
        self.trunk_values.append(float(trunk))
        return True

    @property
    def valid_frames(self):
        return len(self.head_values)

    def finish(self, min_valid_frames=10, min_valid_ratio=0.5):
        if self.total_frames <= 0:
            raise CalibrationError("No calibration frames were processed.")
        if self.valid_frames < min_valid_frames:
            raise CalibrationError(
                f"Insufficient valid frames: {self.valid_frames}/{self.total_frames} "
                f"(minimum {min_valid_frames})"
            )
        if self.valid_frames / self.total_frames < min_valid_ratio:
            raise CalibrationError(
                f"Valid-frame ratio is too low: {self.valid_frames}/{self.total_frames} "
                f"(minimum {min_valid_ratio:.0%})"
            )

        head_baseline = statistics.median(self.head_values)
        trunk_baseline = statistics.median(self.trunk_values)
        return CalibrationBaseline(
            head_forward_ratio=head_baseline,
            trunk_angle=trunk_baseline,
            valid_frames=self.valid_frames,
            total_frames=self.total_frames,
            head_mad=_median_absolute_deviation(self.head_values, head_baseline),
            trunk_mad=_median_absolute_deviation(self.trunk_values, trunk_baseline),
        )


def calibrated_value(value, baseline):
    """Return deterioration from the baseline; improvements are clamped to zero."""
    if not (_is_finite(value) and _is_finite(baseline)):
        return None
    return max(0.0, float(value) - float(baseline))


def _median_absolute_deviation(values, center):
    return statistics.median(abs(value - center) for value in values)


def _is_finite(value):
    return value is not None and math.isfinite(float(value))

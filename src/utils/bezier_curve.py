import json
import bezier
import numpy as np


# Read curve data from the scene.blend file
# -----------------------------------------

class MultipleSegmentsBezierCurve(object):
    def __init__(self, curve_data):
        self.n_segments = len(curve_data) - 1
        self.raw_segments = [
            np.array([
                curve_data[i]["point"],
                curve_data[i]["next_handle"],
                curve_data[i + 1]["prev_handle"],
                curve_data[i + 1]["point"]
            ], dtype=np.float32)
            for i in range(self.n_segments)
        ]
        self.curve_segments = [
            bezier.Curve(segment.transpose(), degree=3)
            for segment in self.raw_segments
        ]

        self._segment_lengths = np.array([
            segment.length
            for segment in self.curve_segments
        ], dtype=np.float32)

        self.segment_proportions = self._segment_lengths / np.sum(self._segment_lengths)

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, "r") as file:
            curves = json.load(file)
        return {
            curve_name: MultipleSegmentsBezierCurve(curve_data=curve_data)
            for curve_name, curve_data in curves.items()
        }

    @property
    def length(self):
        return sum(self._segment_lengths)

    def evaluate(self, t: np.array) -> np.array:
        curve_points = []
        start_segment = 0
        for n in range(self.n_segments):
            end_segment = start_segment + self.segment_proportions[n]
            mask = (t >= start_segment) & (t < end_segment)
            if mask.any():
                segment_t = (t[mask] - start_segment) / (end_segment - start_segment)
                curve_points.append(
                    self.curve_segments[n].evaluate_multi(segment_t)
                )
            start_segment = end_segment
        return np.concatenate(curve_points, axis=1)

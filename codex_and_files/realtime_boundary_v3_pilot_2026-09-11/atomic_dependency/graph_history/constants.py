from __future__ import annotations

NUM_GRAPH_NODES = 35
NUM_TIER3_CLASSES = 31
DEFAULT_CAMERA_ID = "001484412812"
DEFAULT_RGB_MEAN = (0.5369, 0.5295, 0.5208)
DEFAULT_RGB_STD = (0.2311, 0.2360, 0.2363)

MODEL_NAMES = {
    "m0": "current_only",
    "m1": "history_no_position",
    "m2": "actual_history",
    "m3": "graph_valid_shuffle",
    "m4": "candidate_no_graph",
    "m5": "graph_oracle",
    "m6": "graph_predicted",
}

RELATION_TO_ID = {"I": 0, "M": 1, "O": 2, "X": 3, "S": 4}
ID_TO_RELATION = {value: key for key, value in RELATION_TO_ID.items()}


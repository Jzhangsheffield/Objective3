"""Phase A multimodal incremental-value experiments built on M2-Direct."""

CONDITIONS = {
    "A0": ("primary",),
    "A1": ("secondary",),
    "A2": ("primary", "secondary"),
    "A3": ("primary", "secondary"),
    "A4": ("primary", "imu"),
    "A5": ("primary", "emg"),
    "A6": ("primary", "emg", "imu"),
    "A7": ("primary", "secondary", "emg", "imu"),
}

TRAINABLE_CONDITIONS = {"A1", "A3", "A4", "A5", "A6", "A7"}

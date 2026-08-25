"""Feature-layout constants for the final MOF and LD50 MINTNN experiments."""

MOF_FEATURE_LAYOUTS = {
    "homology": {
        "flat_width": 3240,
        "model_axes": "9 categories x 120 filtrations x 3 Betti dimensions",
        "source_axes": "9 categories x 3 Betti dimensions x 120 filtrations",
    },
    "facet": {
        "flat_width": 5445,
        "model_axes": "9 categories x 121 filtrations x 5 facet curves",
        "source_axes": "9 categories x 5 facet curves x 121 filtrations",
    },
    "forman": {
        "flat_width": 21600,
        "model_axes": "9 categories x 120 filtrations x 20 Forman channels",
        "source_axes": "9 categories x 120 filtrations x 20 Forman channels",
    },
    "lap": {
        "flat_width": 10800,
        "model_axes": "9 categories x 120 filtrations x 10 channels",
        "source_axes": "9 categories x 120 filtrations x 10 channels",
        "note": "Final callZeroLap version with zero-filled Laplacian channels for Call plus Call Betti1/Betti2.",
    },
}

LD50_FEATURE_LAYOUTS = {
    "homology": {"raw_shape": "100 x 30 x 2", "model_axes": "30 x 100 x 2"},
    "facet": {"raw_shape": "100 x 30 x 2", "model_axes": "30 x 100 x 2"},
    "lap": {"raw_shape": "100 x 30 x 8", "model_axes": "30 x 100 x 8"},
    "forman": {"raw_shape": "100 x 30 x 20", "model_axes": "30 x 100 x 20"},
    "curvature": {
        "single_direction_raw_shape": "49 x 30 x 10",
        "bidirectional_raw_shape": "49 x 30 x 20",
        "model_axes": "30 x 49 x channels",
    },
}

LD50_PAIR_ELEMENTS = {
    "base_elements": ["H", "C", "N", "O"],
    "all_elements": ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"],
    "rule": "a in base_elements, b in all_elements, element_order[a] < element_order[b]",
    "pair_count": 30,
}

FIELD_SPACING = 1
SECTION_SPACING = 4

PARAMETER_CONFIG = {
    # CLAHE
    "clahe_clip": {
        "command": "clahe",
        "default": 2.0,
        "min": 1.0,
        "max": 10.0,
    },
    "clahe_tile": {
        "command": "clahe",
        "default": 8,
        "min": 2,
        "max": 32,
    },

    # Gaussian (example defaults)
    "gauss_kernel": {
        "command": "toggle_blur",
        "default": 5,
        "min": 1,
        "max": 31,
    },
    "gauss_sigma": {
        "command": "toggle_blur",
        "default": 1.0,
        "min": 0.1,
        "max": 10.0,
    },

    # Canny
    "canny_t1": {
        "command": "toggle_canny",
        "default": 50,
        "min": 0,
        "max": 500,
    },
    "canny_t2": {
        "command": "toggle_canny",
        "default": 150,
        "min": 0,
        "max": 500,
    },

    # Hough
    "hough_dp": {
        "command": "toggle_hough",
        "default": 1.2,
        "min": 1.0,
        "max": 3.0,
    },
    "hough_mindist": {
        "command": "toggle_hough",
        "default": 50,
        "min": 1,
        "max": 500,
    },
    "hough_param1": {
        "command": "toggle_hough",
        "default": 100,
        "min": 1,
        "max": 500,
    },
    "hough_param2": {
        "command": "toggle_hough",
        "default": 30,
        "min": 1,
        "max": 200,
    },
    "hough_minr": {
        "command": "toggle_hough",
        "default": 0,
        "min": 0,
        "max": 500,
    },
    "hough_maxr": {
        "command": "toggle_hough",
        "default": 0,
        "min": 0,
        "max": 500,
    },
}
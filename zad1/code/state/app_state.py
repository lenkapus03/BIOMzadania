import json
import os

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))

class AppState:
    def __init__(self):
        self.texture_id = None
        self.current_image_path = None
        self.active_circle = "iris"
        self.current_record = None
        self.ground_truth_circles = None

        self.circle_params = {
            "iris": {},
            "pupil": {},
            "upper_lid": {},
            "lower_lid": {}
        }
        self._load_all_configs()

    def _load_all_configs(self):
        for circle in self.circle_params:
            path = os.path.join(CONFIG_DIR, f"{circle}_config.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.circle_params[circle] = json.load(f)
import json
import os

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))


class AppState:
    """Uchováva globálny stav aplikácie — aktívnu kružnicu, parametre, aktuálny záznam a textúru."""

    def __init__(self):
        self.texture_id = None           # ID aktuálnej DPG textúry
        self.current_image_path = None   # Cesta k aktuálne načítanému obrazku
        self.active_circle = "iris"      # Aktuálne vybraná kružnica (iris/pupil/upper_lid/lower_lid)
        self.current_record = None       # Aktuálny ImageRecord z CSV
        self.ground_truth_circles = None # Ground truth kružnice pre aktuálny záznam

        # Uložené parametre predspracovania a detekcie pre každú kružnicu
        self.circle_params = {
            "iris": {},
            "pupil": {},
            "upper_lid": {},
            "lower_lid": {}
        }
        # Načítaj uložené konfigurácie pri štarte aplikácie
        self._load_all_configs()

    def _load_all_configs(self):
        """Načíta uložené JSON konfigurácie pre každú kružnicu z config priečinka."""
        for circle in self.circle_params:
            path = os.path.join(CONFIG_DIR, f"{circle}_config.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.circle_params[circle] = json.load(f)
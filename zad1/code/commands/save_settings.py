import json
import os
import dearpygui.dearpygui as dpg
from zad1.code.commands.image_command import ImageCommand
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS, CANVAS_TAGS

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))

class SaveSettingsCommand(ImageCommand):
    def __init__(self, processor, renderer, state):
        super().__init__(processor, renderer)
        self.state = state

    def execute(self, sender=None, app_data=None, user_data=None):
        current = self.state.active_circle

        self.state.circle_params[current] = {
            tag: dpg.get_value(tag)
            for tag in PARAMETER_CONFIG
        }
        for tag in TOGGLE_TAGS:
            self.state.circle_params[current][tag] = dpg.get_value(tag)
        for tag in CANVAS_TAGS:
            val = dpg.get_value(tag)
            print(f"Saving canvas tag {tag} = {val}")
            self.state.circle_params[current][tag] = val

        os.makedirs(CONFIG_DIR, exist_ok=True)
        for circle, params in self.state.circle_params.items():
            path = os.path.join(CONFIG_DIR, f"{circle}_config.json")
            with open(path, "w") as f:
                json.dump(params, f, indent=2)
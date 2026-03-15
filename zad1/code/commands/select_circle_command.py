import json
import os
from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

from zad1.code.core.helpers import apply_params
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS, CANVAS_TAGS

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))

class SelectCircleCommand(ImageCommand):
    def __init__(self, processor, renderer, state):
        super().__init__(processor, renderer)
        self.state = state

    def execute(self, sender=None, app_data=None, user_data=None):
        old = self.state.active_circle
        self.state.circle_params[old] = {
            tag: dpg.get_value(tag)
            for tag in PARAMETER_CONFIG
        }
        for tag in TOGGLE_TAGS:
            self.state.circle_params[old][tag] = dpg.get_value(tag)
        for tag in CANVAS_TAGS:
            self.state.circle_params[old][tag] = dpg.get_value(tag)

        self.state.active_circle = app_data

        saved = self.state.circle_params[app_data]
        apply_params(saved, self.processor, self.renderer)
        self.refresh(apply_processor=False)
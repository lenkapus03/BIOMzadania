from zad1.code.commands.base_command import Command
from zad1.code.commands.helpers import update_texture

import dearpygui.dearpygui as dpg

class UpdateCanvasCommand(Command):
    def __init__(self, processor, state):
        self.processor = processor
        self.state = state

    def execute(self, sender=None, app_data=None, user_data=None):
        # just change the canvas size
        w = dpg.get_value("canvas_width")
        h = dpg.get_value("canvas_height")

        self.processor.canvas_width = w
        self.processor.canvas_height = h

        # generate texture from existing processed image
        data = self.processor.get_scaled_texture()
        if data is not None:
            update_texture(self.state, data, w, h)
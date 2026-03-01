import dearpygui.dearpygui as dpg
from zad1.code.commands.image_command import ImageCommand

class UpdateCanvasCommand(ImageCommand):

    def execute(self, sender=None, app_data=None, user_data=None):
        # Update canvas size in processor
        self.processor.canvas_width = dpg.get_value("canvas_width")
        self.processor.canvas_height = dpg.get_value("canvas_height")

        # Only regenerate texture
        data = self.processor.get_scaled_texture()
        if data is not None:
            from zad1.code.commands.helpers import update_texture
            update_texture(
                self.state,
                data,
                self.processor.canvas_width,
                self.processor.canvas_height
            )
from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

class UpdateCanvasCommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.renderer.canvas_width = int(dpg.get_value("canvas_width"))
        self.renderer.canvas_height = int(dpg.get_value("canvas_height"))

        self.refresh(apply_processor=True)
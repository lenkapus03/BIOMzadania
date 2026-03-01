from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

class CLAHECommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.clip_limit = float(dpg.get_value("clahe_clip"))
        self.processor.grid_size = int(dpg.get_value("clahe_tile"))
        self.refresh()
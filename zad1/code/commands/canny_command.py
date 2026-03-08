from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg


class CannyCommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.canny_threshold_1 = int(dpg.get_value("threshold_1"))
        self.processor.canny_threshold_2 = float(dpg.get_value("threshold_2"))

        self.refresh()
from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

class GaussianBlurCommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.gauss_kernel = int(dpg.get_value("gauss_kernel"))
        self.processor.gauss_sigma = float(dpg.get_value("gauss_sigma"))

        self.refresh()
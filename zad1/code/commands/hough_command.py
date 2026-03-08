from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

class HoughCommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.hough_dp = float(dpg.get_value("hough_dp"))
        self.processor.hough_mindist = int(dpg.get_value("hough_mindist"))
        self.processor.hough_param1 = int(dpg.get_value("hough_param1"))
        self.processor.hough_param2 = int(dpg.get_value("hough_param2"))
        self.processor.hough_minr = int(dpg.get_value("hough_minr"))
        self.processor.hough_maxr = int(dpg.get_value("hough_maxr"))
        self.refresh()
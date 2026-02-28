from zad1.code.commands.base_command import Command
from zad1.code.commands.helpers import update_texture

class HistogramEqualizationCommand(Command):
    def __init__(self, processor, state):
        self.processor = processor
        self.state = state

    def execute(self, sender=None, app_data=None, user_data=None):
        # update processor setting
        self.processor.use_histogram_eq = bool(app_data)

        # regenerate the image with all current settings
        self.processor.apply()
        data = self.processor.get_scaled_texture()
        if data is not None:
            update_texture(
                self.state,
                data,
                self.processor.canvas_width,
                self.processor.canvas_height
            )
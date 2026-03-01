from zad1.code.commands.image_command import ImageCommand

class HistogramEqualizationCommand(ImageCommand):

    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.use_histogram_eq = bool(app_data)

        self.refresh()
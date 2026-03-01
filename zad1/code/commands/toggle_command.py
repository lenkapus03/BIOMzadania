from zad1.code.commands.image_command import ImageCommand


class ToggleCommand(ImageCommand):
    def __init__(self, processor, renderer, attr_name):
        super().__init__(processor, renderer)
        self.attr_name = attr_name

    def execute(self, sender=None, app_data=None, user_data=None):
        setattr(self.processor, self.attr_name, bool(app_data))
        self.refresh()
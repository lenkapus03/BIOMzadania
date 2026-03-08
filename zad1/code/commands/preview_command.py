from zad1.code.commands.image_command import ImageCommand

class PreviewOriginalCommand(ImageCommand):
    def __init__(self, processor, renderer, show_original: bool):
        super().__init__(processor, renderer)
        self.show_original = show_original

    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.preview_original = bool(app_data)
        self.refresh(apply_processor=False)
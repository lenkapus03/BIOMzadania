from abc import ABC

from zad1.code.commands.base_command import Command
from zad1.code.commands.helpers import update_texture

class ImageCommand(Command, ABC):
    def __init__(self, processor, state):
        self.processor = processor
        self.state = state

    def refresh(self):
        # First, reapply transformations according to current processor settings
        self.processor.apply()
        # Then update the texture
        data = self.processor.get_scaled_texture()
        if data is not None:
            update_texture(
                self.state,
                data,
                self.processor.canvas_width,
                self.processor.canvas_height
            )
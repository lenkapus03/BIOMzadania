from abc import ABC
from zad1.code.commands.base_command import Command

class ImageCommand(Command, ABC):
    def __init__(self, processor, renderer):
        self.processor = processor
        self.renderer = renderer

    def refresh(self, apply_processor=True):
        self.renderer.refresh_texture(apply_processor)
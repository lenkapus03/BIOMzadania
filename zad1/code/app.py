import os
import dearpygui.dearpygui as dpg

from zad1.code.commands.dispatcher import CommandDispatcher
from zad1.code.commands.histogram_equalization import HistogramEqualizationCommand
from zad1.code.commands.update_canvas import UpdateCanvasCommand
from zad1.code.core.image_manager import ImageManager
from zad1.code.core.image_processor import ImageProcessor
from zad1.code.state.app_state import AppState
from zad1.code.ui.controls_window import ControlsWindow
from zad1.code.ui.image_window import ImageWindow
from zad1.code.data import load_data


class Application:
    def __init__(self):
        # Core objects
        self.state = AppState()
        self.image_manager = ImageManager()
        self.processor = None
        self.dispatcher = CommandDispatcher()

        # Fonts
        self.fonts = (None, None)

        # UI windows
        self.controls_window = None
        self.image_window = None

        # Initialize app
        self._load_first_image()
        self._init_processor()
        self._register_commands()

    # Initialization helpers
    def _load_first_image(self):
        if not load_data.RANDOM_RECORDS:
            return

        first_record = load_data.RANDOM_RECORDS[0]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.normpath(os.path.join(current_dir, "..", "data"))
        full_path = os.path.normpath(os.path.join(data_folder, first_record.image.replace("\\", "/")))

        self.image_manager.load_image(full_path)

    def _init_processor(self):
        self.processor = ImageProcessor(
            original_image=self.image_manager.original_image
        )

    def _create_initial_texture(self):
        if self.processor is None or self.processor.processed_image is None:
            return

        width = self.processor.canvas_width
        height = self.processor.canvas_height
        if width == 0 or height == 0:
            return

        canvas_data = self.processor.get_scaled_texture()
        if canvas_data is None:
            # fallback to blank RGBA
            canvas_data = [0.0] * (width * height * 4)

        # Only create the registry once
        if not hasattr(self, "_texture_registry"):
            self._texture_registry = dpg.add_texture_registry(show=False)

        # Remove old texture if exists
        if hasattr(self.state, "texture_id") and dpg.does_item_exist(self.state.texture_id):
            dpg.delete_item(self.state.texture_id)

        # Add the new texture id into state
        self.state.texture_id = dpg.add_dynamic_texture(
            width=width,
            height=height,
            default_value=canvas_data,
            parent=self._texture_registry
        )

    def _register_commands(self):
        self.dispatcher.register("update_canvas", UpdateCanvasCommand(self.processor, self.state))
        self.dispatcher.register("toggle_histogram", HistogramEqualizationCommand(self.processor, self.state))

    # UI helpers
    def _setup_fonts(self):
        with dpg.font_registry():
            large_font = dpg.add_font("C:/Windows/Fonts/arial.ttf", 12)
            small_font = dpg.add_font("C:/Windows/Fonts/arial.ttf", 10)
        self.fonts = (large_font, small_font)

    def _create_windows(self):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()

        controls_width = int(viewport_width * 0.3)
        image_width = int(viewport_width * 0.7)

        self.controls_window = ControlsWindow(
            dispatcher=self.dispatcher,
            fonts=self.fonts,
            width=controls_width,
            height=viewport_height
        )

        self.image_window = ImageWindow(
            state=self.state,
            width=image_width,
            height=viewport_height,
            pos_x=controls_width
        )

    def _build_ui(self):
        self._setup_fonts()
        self._create_windows()
        self.controls_window.build()
        self.image_window.build()

    # Run app
    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Zad 1")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()

        self._create_initial_texture()
        self._build_ui()

        dpg.start_dearpygui()
        dpg.destroy_context()
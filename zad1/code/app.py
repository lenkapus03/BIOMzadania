import dearpygui.dearpygui as dpg
import os

from zad1.code.commands.dispatcher import CommandDispatcher
from zad1.code.commands.update_canvas import UpdateCanvasCommand
from zad1.code.core.image_manager import ImageManager
from zad1.code.state.app_state import AppState
from zad1.code.ui.controls_window import ControlsWindow
from zad1.code.ui.image_window import ImageWindow
from zad1.code.data import load_data


class Application:
    def __init__(self):
        self.state = AppState()
        self.image_manager = ImageManager()
        self.dispatcher = CommandDispatcher()
        self._register_commands()

        self.controls_window = None
        self.image_window = None

    def _register_commands(self):
        self.dispatcher.register(
            "update_canvas",
            UpdateCanvasCommand(self.image_manager, self.state)
        )

    # UI build methods
    def _build_ui(self):
        # 1. Load first random image if available
        self._load_initial_image()

        # 2. Setup fonts
        self.fonts = self._setup_fonts()

        # 3. Create windows with correct dimensions
        self._create_windows()

        # 4. Build the windows
        self.controls_window.build()
        self.image_window.build()

    def _load_initial_image(self):
        if not load_data.RANDOM_RECORDS:
            return

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.normpath(os.path.join(current_dir, "..", "data"))
        first_record = load_data.RANDOM_RECORDS[0]

        # Resolve image path
        rel_path = first_record.image.replace("\\", "/")
        full_path = os.path.normpath(os.path.join(data_folder, rel_path))

        # Load into ImageManager
        self.image_manager.load_image(full_path)

        # Create initial texture
        img_h, img_w = self.image_manager.original_image.shape[:2]
        with dpg.texture_registry():
            tex_data = self.image_manager.get_scaled_texture(img_w, img_h)
            self.state.texture_id = dpg.add_dynamic_texture(
                width=img_w,
                height=img_h,
                default_value=tex_data if tex_data is not None else [0.0] * (img_w * img_h * 4)
            )

    def _setup_fonts(self):
        with dpg.font_registry():
            large_font = dpg.add_font("C:/Windows/Fonts/arial.ttf", 12)
            small_font = dpg.add_font("C:/Windows/Fonts/arial.ttf", 10)
        return large_font, small_font

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

    # Run APP
    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Zad 1")

        # Setup viewport
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()

        # Build UI
        self._build_ui()

        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    app = Application()
    app.run()
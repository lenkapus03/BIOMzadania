import os
import dearpygui.dearpygui as dpg

from zad1.code.commands.canny_command import CannyCommand
from zad1.code.commands.clahe_command import CLAHECommand
from zad1.code.commands.dispatcher import CommandDispatcher
from zad1.code.commands.evaluate_command import EvaluateCommand
from zad1.code.commands.gaussian_blur_command import GaussianBlurCommand
from zad1.code.commands.hough_command import HoughCommand
from zad1.code.commands.save_settings import SaveSettingsCommand
from zad1.code.commands.select_circle_command import SelectCircleCommand
from zad1.code.commands.toggle_command import ToggleCommand
from zad1.code.commands.toggle_renderer_command import ToggleRendererCommand
from zad1.code.commands.update_canvas_command import UpdateCanvasCommand
from zad1.code.core.helpers import apply_params
from zad1.code.core.image_manager import ImageManager
from zad1.code.core.image_processor import ImageProcessor
from zad1.code.core.renderer import Renderer
from zad1.code.state.app_state import AppState
from zad1.code.ui.controls_window import ControlsWindow
from zad1.code.ui.image_window import ImageWindow
from zad1.code.data import load_data


class Application:
    def __init__(self):
        # Hlavné objekty
        self.state = AppState()
        self.image_manager = ImageManager()
        self.processor = None
        self.renderer = None
        self.dispatcher = CommandDispatcher()

        # Fonts
        self.fonts = (None, None)

        # UI windows
        self.controls_window = None
        self.image_window = None

        # Inicializuj appku
        self._load_first_image()
        self._init_processor()
        self._init_renderer()
        self._register_commands()

    def _load_first_image(self):
        if not load_data.RANDOM_RECORDS:
            return

        first_record = load_data.RANDOM_RECORDS[0]
        self.state.current_record = first_record
        self.state.ground_truth_circles = self._record_to_gt(first_record)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.normpath(os.path.join(current_dir, "..", "data"))
        full_path = os.path.normpath(os.path.join(data_folder, first_record.image.replace("\\", "/")))

        self.image_manager.load_image(full_path)

    def _record_to_gt(self, record):
        return {
            "pupil": (record.center_x_1, record.center_y_1, record.polomer_1),
            "iris": (record.center_x_2, record.center_y_2, record.polomer_2),
            "lower_lid": (record.center_x_3, record.center_y_3, record.polomer_3),
            "upper_lid": (record.center_x_4, record.center_y_4, record.polomer_4),
        }

    def _init_processor(self):
        self.processor = ImageProcessor(
            original_image=self.image_manager.original_image
        )

    def _init_renderer(self):
        self.renderer = Renderer(
            processor=self.processor,
            state=self.state
        )

    def _create_initial_texture(self):
        if self.processor is None or self.processor.processed_image is None:
            return

        width = self.renderer.canvas_width
        height = self.renderer.canvas_height
        if width == 0 or height == 0:
            return

        result = self.renderer.get_scaled_texture()
        if result is None:
            canvas_data = [0.0] * (width * height * 4)
        else:
            canvas_data, width, height = result  # rozbaľ tuple

        if not hasattr(self, "_texture_registry"):
            self._texture_registry = dpg.add_texture_registry(show=False)

        if hasattr(self.state, "texture_id") and dpg.does_item_exist(self.state.texture_id):
            dpg.delete_item(self.state.texture_id)

        self.state.texture_id = dpg.add_dynamic_texture(
            width=width,
            height=height,
            default_value=canvas_data,
            parent=self._texture_registry
        )

    def _register_commands(self):
        self.dispatcher.register("update_canvas", UpdateCanvasCommand(self.processor, self.renderer))

        self.dispatcher.register("use_histogram_eq", ToggleCommand(self.processor, self.renderer, "use_histogram_eq"))

        self.dispatcher.register("use_clahe", ToggleCommand(self.processor, self.renderer, "use_clahe"))
        self.dispatcher.register("clahe", CLAHECommand(self.processor, self.renderer))

        self.dispatcher.register("use_blur", ToggleCommand(self.processor, self.renderer, "use_blur"))
        self.dispatcher.register("gaussian_blur", GaussianBlurCommand(self.processor, self.renderer))

        self.dispatcher.register("use_canny", ToggleCommand(self.processor, self.renderer, "use_canny"))
        self.dispatcher.register("canny", CannyCommand(self.processor, self.renderer))

        self.dispatcher.register("show_hough", ToggleCommand(self.processor, self.renderer, "show_hough"))
        self.dispatcher.register("hough", HoughCommand(self.processor, self.renderer))

        self.dispatcher.register("select_circle", SelectCircleCommand(self.processor, self.renderer, self.state))
        self.dispatcher.register("show_rejected_circles",ToggleCommand(self.processor, self.renderer, "show_rejected_circles"))
        self.dispatcher.register("save_settings", SaveSettingsCommand(self.processor, self.renderer, self.state))
        self.dispatcher.register("preview_original", ToggleCommand(self.processor, self.renderer, "preview_original"))

        self.dispatcher.register("evaluate", EvaluateCommand(self.processor, self.renderer, self.state))

        self.dispatcher.register("show_ground_truth",ToggleRendererCommand(self.processor, self.renderer, "show_ground_truth"))
        self.dispatcher.register("show_all_circles",ToggleRendererCommand(self.processor, self.renderer, "show_all_circles"))    # UI helpers

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

    def _apply_initial_params(self):
        saved = self.state.circle_params[self.state.active_circle]
        apply_params(saved, self.processor, self.renderer)

    def _build_ui(self):
        self._setup_fonts()
        self._create_windows()
        self.controls_window.build()
        self.image_window.build()
        self._apply_initial_params()

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Zad 1")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()

        self._create_initial_texture()
        self._build_ui()
        self.renderer.refresh_texture(apply_processor=True)

        dpg.start_dearpygui()
        dpg.destroy_context()
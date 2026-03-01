import dearpygui.dearpygui as dpg
from zad1.code.ui.ui_helpers import add_slider, add_checkbox, add_section_separator
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG

class ControlsWindow:
    def __init__(self, dispatcher, fonts, width, height):
        self.dispatcher = dispatcher
        self.large_font, self.small_font = fonts
        self.width = width
        self.height = height

    def build(self):
        with dpg.window(label="Controls", width=self.width, height=self.height, pos=(0,0)):
            self._build_view_section()
            self._add_section_separator()

            self._build_on_off_section()
            self._add_section_separator()

            self._build_parameter_sections()
            self._add_section_separator()

            self._build_bottom_buttons()

    # Sections
    def _build_view_section(self):
        title = dpg.add_text("Zobrazenie")
        dpg.bind_item_font(title, self.large_font)

        self._add_slider("Canvas width", "canvas_width", "update_canvas", 320, 320, 1600)
        self._add_slider("Canvas height", "canvas_height", "update_canvas", 280, 280, 1200)

    def _build_on_off_section(self):
        title = dpg.add_text("ON / OFF")
        dpg.bind_item_font(title, self.large_font)

        # Map labels to processor attributes
        toggles = {
            "Histogram equation": "use_histogram_eq",
            "CLAHE": "use_clahe",
            "Gaussian blur": "use_blur",
            "Show canny edges": "show_canny",
            "Hough circles": "show_hough"
        }

        with dpg.group(horizontal=True):
            with dpg.group():
                for label, attr in list(toggles.items())[:3]:
                    self._add_checkbox(label, attr, command_name=attr)
            dpg.add_spacer(width=20)
            with dpg.group():
                for label, attr in list(toggles.items())[3:]:
                    self._add_checkbox(label, attr, command_name=attr)

    def _build_parameter_sections(self):
        sections = [
            ("CLAHE", ["clahe_clip", "clahe_tile"]),
            ("Gaussian Blur", ["gauss_kernel", "gauss_sigma"]),
            ("Canny", ["canny_t1", "canny_t2"]),
            (
                "HoughCircles",
                [
                    "hough_dp", "hough_mindist", "hough_param1", "hough_param2", "hough_minr", "hough_maxr"
                ],
            ),
        ]

        for section_title, param_tags in sections:
            title = dpg.add_text(section_title)
            dpg.bind_item_font(title, self.large_font)

            for tag in param_tags:
                cfg = PARAMETER_CONFIG[tag]
                self._add_slider(
                    label=tag.replace("_", " ").title(),
                    tag=tag,
                    command_name=cfg["command"],
                    default=cfg["default"],
                    min_v=cfg["min"],
                    max_v=cfg["max"]
                )

            self._add_section_separator()

    def _build_bottom_buttons(self):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset defaults", width=120, callback=self._global_callback, user_data="reset_defaults")
            dpg.add_button(label="Export settings", width=120, callback=self._global_callback, user_data="export_settings")

    # Helpers
    def _add_slider(self, label, tag, command_name=None, default=0, min_v=0, max_v=2000):
        add_slider(
            label=label,
            tag=tag,
            small_font=self.small_font,
            callback=self._global_callback if command_name else None,
            default=default,
            min_v=min_v,
            max_v=max_v
        )
        if command_name:
            dpg.set_item_user_data(tag, command_name)

    def _add_checkbox(self, label, processor_attr, command_name="use_histogram_eq"):
        add_checkbox(
            label=label,
            tag=processor_attr,
            small_font=self.small_font,
            callback=lambda s, a, u: self.dispatcher.execute(command_name, s, a, u),
            user_data=processor_attr
        )

    def _add_section_separator(self):
        add_section_separator()

    def _global_callback(self, sender, app_data, user_data):
        command_name = dpg.get_item_user_data(sender)  # ← always the command name
        self.dispatcher.execute(command_name, sender, app_data, user_data)
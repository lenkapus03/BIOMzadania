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
        with dpg.window(
            label="Controls",
            width=self.width,
            height=self.height,
            pos=(0, 0),
        ):
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

        labels = [
            ("Histogram equation", "toggle_histogram"),
            ("CLAHE", "clahe"),
            ("Gaussian blur", "toggle_blur"),
            ("Show canny edges", "toggle_canny"),
            ("Hough circles", "toggle_hough"),
        ]

        with dpg.group(horizontal=True):
            with dpg.group():
                for text, cmd in labels[:3]:
                    self._add_checkbox(text, cmd)

            dpg.add_spacer(width=20)

            with dpg.group():
                for text, cmd in labels[3:]:
                    self._add_checkbox(text, cmd)

    def _build_parameter_sections(self):
        sections = [
            ("CLAHE", ["clahe_clip", "clahe_tile"]),
            ("Gaussian Blur", ["gauss_kernel", "gauss_sigma"]),
            ("Canny", ["canny_t1", "canny_t2"]),
            (
                "HoughCircles",
                [
                    "hough_dp",
                    "hough_mindist",
                    "hough_param1",
                    "hough_param2",
                    "hough_minr",
                    "hough_maxr",
                ],
            ),
        ]

        for title_text, tags in sections:
            title = dpg.add_text(title_text)
            dpg.bind_item_font(title, self.large_font)

            for tag in tags:
                config = PARAMETER_CONFIG[tag]

                self._add_slider(
                    label=tag.replace("_", " ").title(),
                    tag=tag,
                    command_name=config["command"],
                    default=config["default"],
                    min_v=config["min"],
                    max_v=config["max"],
                )

            self._add_section_separator()

    def _build_bottom_buttons(self):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Reset defaults",
                width=120,
                callback=self._global_callback,
                user_data="reset_defaults",
            )
            dpg.add_button(
                label="Export settings",
                width=120,
                callback=self._global_callback,
                user_data="export_settings",
            )

    # UI helper wrappers
    def _add_slider(self, label, tag, command_name=None, default=0, min_v=0, max_v=2000):
        add_slider(
            label=label,
            tag=tag,
            small_font=self.small_font,
            callback=self._global_callback if command_name else None,
            default=default,
            min_v=min_v,
            max_v=max_v,
        )

        if command_name:
            dpg.set_item_user_data(tag, command_name)

    def _add_checkbox(self, label, command_name):
        tag = f"chk_{label.replace(' ', '_')}"
        add_checkbox(label, tag, self.small_font)
        dpg.set_item_callback(tag, self._global_callback)
        dpg.set_item_user_data(tag, command_name)

    def _add_section_separator(self):
        add_section_separator()

    # command dispatcher
    def _global_callback(self, sender, app_data, user_data):
        if user_data:
            self.dispatcher.execute(user_data, sender, app_data)
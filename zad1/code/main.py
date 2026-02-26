import dearpygui.dearpygui as dpg
from win32api import GetSystemMetrics

FIELD_SPACING = 1
SECTION_SPACING = 5

def get_resolution():
    return GetSystemMetrics(0), GetSystemMetrics(1)

def add_spacing(height):
    dpg.add_spacer(height=height)

def add_slider(label, tag, small_font, default=280, min_v=1, max_v=900):
    slider = dpg.add_slider_int(
        label=label,
        tag=tag,
        default_value=default,
        min_value=min_v,
        max_value=max_v,
    )
    dpg.bind_item_font(slider, small_font)
    add_spacing(FIELD_SPACING)

def add_checkbox(label, tag, small_font):
    checkbox = dpg.add_checkbox(label=label, tag=tag)
    dpg.bind_item_font(checkbox, small_font)
    add_spacing(FIELD_SPACING)

def add_section_separator():
    dpg.add_separator()
    add_spacing(SECTION_SPACING)

def main():
    dpg.create_context()

    win_width, win_height = get_resolution()
    dpg.create_viewport(
        title="Zad1",
        width=win_width,
        height=win_height
    )

    with dpg.font_registry():
        large_font = dpg.add_font("C:/Windows/Fonts/arial.ttf", 14)
        small_font = dpg.add_font("C:/Windows/Fonts/arial.ttf", 12)

    with dpg.window(
        label="Controls",
        width=int(0.3 * win_width),
        height=win_height,
        pos=(10, 10),
    ):

        # Zobrazenie
        title = dpg.add_text("Zobrazenie")
        dpg.bind_item_font(title, large_font)
        add_spacing(FIELD_SPACING)

        add_slider("Canvas width", "canvas_width", small_font, 320, 1, 1600)
        add_slider("Canvas height", "canvas_height", small_font, 280, 1, 900)

        add_section_separator()

        # ---- ON/OFF ----
        title = dpg.add_text("ON OFF")
        dpg.bind_item_font(title, large_font)
        add_spacing(FIELD_SPACING)

        labels = [
            "Histogram equation",
            "CLAHE",
            "Gaussian blur",
            "Show canny edges",
            "Hough circles overlay",
        ]

        with dpg.group(horizontal=True):

            with dpg.group():
                for i in range(3):
                    add_checkbox(f"{labels[i]}", f"checkbox_{i}", small_font)

            dpg.add_spacer(width=int(0.07 * win_width))

            with dpg.group():
                for i in range(3, 5):
                    add_checkbox(f"{labels[i]}", f"checkbox_{i}", small_font)

        add_section_separator()

        # CLAHE
        title = dpg.add_text("CLAHE")
        dpg.bind_item_font(title, large_font)
        add_spacing(FIELD_SPACING)

        add_slider("Clip Limit", "clahe_clip", small_font)
        add_slider("Tile Grid Size", "clahe_tile", small_font)

        add_section_separator()

        # Gaussian Blur
        title = dpg.add_text("Gaussian Blur")
        dpg.bind_item_font(title, large_font)
        add_spacing(FIELD_SPACING)

        add_slider("Kernel Size", "gauss_kernel", small_font)
        add_slider("Sigma", "gauss_sigma", small_font)

        add_section_separator()

        # Canny
        title = dpg.add_text("Canny")
        dpg.bind_item_font(title, large_font)
        add_spacing(FIELD_SPACING)

        add_slider("Threshold 1", "canny_t1", small_font)
        add_slider("Threshold 2", "canny_t2", small_font)

        add_section_separator()

        # HoughCircles
        title = dpg.add_text("HoughCircles")
        dpg.bind_item_font(title, large_font)
        add_spacing(FIELD_SPACING)

        add_slider("dp", "hough_dp", small_font)
        add_slider("MinDist", "hough_mindist", small_font)
        add_slider("Param1", "hough_param1", small_font)
        add_slider("Param2", "hough_param2", small_font)
        add_slider("MinRadius", "hough_minr", small_font)
        add_slider("MaxRadius", "hough_maxr", small_font)

        add_spacing(SECTION_SPACING)
        dpg.add_button(label="Reset defaults")
        dpg.add_button(label="Export settings")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
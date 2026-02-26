import os
import dearpygui.dearpygui as dpg
from win32api import GetSystemMetrics
from image import ImageManager
from zad1.code import load_data

FIELD_SPACING = 1
SECTION_SPACING = 4


class AppState:
    texture_id = None


state = AppState()
image_manager = ImageManager()


def update_canvas_callback():
    if not dpg.does_item_exist("canvas_width") or not dpg.does_item_exist("canvas_height"):
        return

    w = dpg.get_value("canvas_width")
    h = dpg.get_value("canvas_height")

    new_data = image_manager.get_scaled_texture(w, h)
    if new_data is None:
        return

    # create new texture
    with dpg.texture_registry():
        new_texture_id = dpg.add_dynamic_texture(width=w, height=h, default_value=new_data)

    # bind texture to widget
    if dpg.does_item_exist("displayed_image_widget"):
        dpg.configure_item("displayed_image_widget", texture_tag=new_texture_id, width=w, height=h)

    # delete old texture
    if state.texture_id is not None and dpg.does_item_exist(state.texture_id):
        dpg.delete_item(state.texture_id)

    # save current texture id
    state.texture_id = new_texture_id


def get_resolution():
    return GetSystemMetrics(0), GetSystemMetrics(1)


def add_spacing(height):
    dpg.add_spacer(height=height)


def add_slider(label, tag, small_font, callback, default=280, min_v=1, max_v=900):
    slider = dpg.add_slider_int(
        label=label, tag=tag, default_value=default,
        min_value=min_v, max_value=max_v, callback=callback
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

    # Setup app window view
    win_w, win_h = get_resolution()
    win_w, win_h = int(0.9 * win_w), int(0.9 * win_h)
    image_win_w = int(0.7 * win_w) - 30
    dpg.create_viewport(title="Zad1", width=win_w, height=win_h)

    # Load image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.normpath(os.path.join(current_dir, "..", "data"))
    csv_path = os.path.join(data_folder, "iris_annotation.csv")
    load_data.load_random_records(csv_path, n=100)

    img_w, img_h = 320, 280
    if load_data.RANDOM_RECORDS:
        rel_path = load_data.RANDOM_RECORDS[0].image.replace("\\", "/")
        full_path = os.path.normpath(os.path.join(data_folder, rel_path))
        image_manager.load_image(full_path)
        img_h, img_w = image_manager.original_image.shape[:2]

    # Fonts
    with dpg.font_registry():
        f_path = "C:/Windows/Fonts/arial.ttf"
        large_font = dpg.add_font(f_path, 16) if os.path.exists(f_path) else dpg.default_font()
        small_font = dpg.add_font(f_path, 12) if os.path.exists(f_path) else dpg.default_font()

    # Initial texture creation
    with dpg.texture_registry():
        tex_data = image_manager.get_scaled_texture(img_w, img_h)
        state.texture_id = dpg.add_dynamic_texture(
            width=img_w, height=img_h,
            default_value=tex_data if tex_data is not None else [0.0] * (img_w * img_h * 4)
        )

    # Window: Controls
    with dpg.window(label="Controls", width=int(0.3 * win_w), height=win_h, pos=(0, 0)):
        title = dpg.add_text("Zobrazenie")
        dpg.bind_item_font(title, large_font)

        add_slider("Canvas width", "canvas_width", small_font, lambda: update_canvas_callback(), img_w, img_w, 1600)
        add_slider("Canvas height", "canvas_height", small_font, lambda: update_canvas_callback(), img_h, img_h, 1200)

        add_section_separator()
        t_onoff = dpg.add_text("ON OFF")
        dpg.bind_item_font(t_onoff, large_font)

        labels = ["Histogram equation", "CLAHE", "Gaussian blur", "Show canny edges", "Hough circles"]
        with dpg.group(horizontal=True):
            with dpg.group():
                for i in range(3): add_checkbox(labels[i], f"checkbox_{i}", small_font)
            dpg.add_spacer(width=20)
            with dpg.group():
                for i in range(3, 5): add_checkbox(labels[i], f"checkbox_{i}", small_font)

        add_section_separator()

        sections = [
            ("CLAHE", ["clahe_clip", "clahe_tile"]),
            ("Gaussian Blur", ["gauss_kernel", "gauss_sigma"]),
            ("Canny", ["canny_t1", "canny_t2"]),
            ("HoughCircles", ["hough_dp", "hough_mindist", "hough_param1", "hough_param2", "hough_minr", "hough_maxr"])
        ]
        for sec_title, tags in sections:
            st = dpg.add_text(sec_title)
            dpg.bind_item_font(st, large_font)
            for tag in tags: add_slider(tag.replace("_", " ").title(), tag, small_font, None)
            add_section_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset defaults", width=120)
            dpg.add_button(label="Export settings", width=120)

    # Window: view image
    with dpg.window(label="View image", width=image_win_w, height=win_h, pos=(int(0.3 * win_w) + 5, 0)):
        dpg.add_image(state.texture_id, tag="displayed_image_widget")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
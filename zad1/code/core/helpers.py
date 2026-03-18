import dearpygui.dearpygui as dpg
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS, CANVAS_TAGS

def update_texture(state, image_data, width, height):
    if image_data is None:
        return None

    # Delete previous texture if it exists
    if getattr(state, 'texture_id', None) is not None and dpg.does_item_exist(state.texture_id):
        dpg.delete_item(state.texture_id)

    # Create new texture
    with dpg.texture_registry():
        tex_id = dpg.add_dynamic_texture(width=width, height=height, default_value=image_data)

    state.texture_id = tex_id

    # Update displayed image widget if it exists
    if dpg.does_item_exist("displayed_image_widget"):
        dpg.configure_item(
            "displayed_image_widget",
            texture_tag=tex_id,
            width=width,
            height=height
        )

    return tex_id

def apply_params(saved, processor, renderer):
    for tag, cfg in PARAMETER_CONFIG.items():
        value = saved.get(tag, cfg["default"])
        dpg.set_value(tag, value)

    for tag in TOGGLE_TAGS:
        value = saved.get(tag, False)
        dpg.set_value(tag, value)
        setattr(processor, tag, value)

    for tag in CANVAS_TAGS:
        default = 320 if tag == "canvas_width" else 280
        dpg.set_value(tag, saved.get(tag, default))

    renderer.canvas_width = int(saved.get("canvas_width", 320))
    renderer.canvas_height = int(saved.get("canvas_height", 280))

    processor.clip_limit = saved.get("clahe_clip", PARAMETER_CONFIG["clahe_clip"]["default"])
    processor.grid_size = int(saved.get("clahe_tile", PARAMETER_CONFIG["clahe_tile"]["default"]))
    processor.gauss_kernel = int(saved.get("gauss_kernel", PARAMETER_CONFIG["gauss_kernel"]["default"]))
    processor.gauss_sigma = float(saved.get("gauss_sigma", PARAMETER_CONFIG["gauss_sigma"]["default"]))
    processor.canny_threshold_1 = int(saved.get("threshold_1", PARAMETER_CONFIG["threshold_1"]["default"]))
    processor.canny_threshold_2 = int(saved.get("threshold_2", PARAMETER_CONFIG["threshold_2"]["default"]))
    processor.hough_dp = float(saved.get("hough_dp", PARAMETER_CONFIG["hough_dp"]["default"]))
    processor.hough_mindist = int(saved.get("hough_mindist", PARAMETER_CONFIG["hough_mindist"]["default"]))
    processor.hough_param1 = int(saved.get("hough_param1", PARAMETER_CONFIG["hough_param1"]["default"]))
    processor.hough_param2 = int(saved.get("hough_param2", PARAMETER_CONFIG["hough_param2"]["default"]))
    processor.hough_minr = int(saved.get("hough_minr", PARAMETER_CONFIG["hough_minr"]["default"]))
    processor.hough_maxr = int(saved.get("hough_maxr", PARAMETER_CONFIG["hough_maxr"]["default"]))

    processor.apply()

def apply_params_headless(cfg, processor, renderer):
    """Verzia apply_params bez DPG — pre použitie mimo aplikácie (grid search)."""
    from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS

    for tag in TOGGLE_TAGS:
        if tag in cfg:
            setattr(processor, tag, cfg[tag])

    processor.clip_limit = cfg.get("clahe_clip", PARAMETER_CONFIG["clahe_clip"]["default"])
    processor.grid_size = int(cfg.get("clahe_tile", PARAMETER_CONFIG["clahe_tile"]["default"]))
    processor.gauss_kernel = int(cfg.get("gauss_kernel", PARAMETER_CONFIG["gauss_kernel"]["default"]))
    processor.gauss_sigma = float(cfg.get("gauss_sigma", PARAMETER_CONFIG["gauss_sigma"]["default"]))
    processor.canny_threshold_1 = int(cfg.get("threshold_1", PARAMETER_CONFIG["threshold_1"]["default"]))
    processor.canny_threshold_2 = int(cfg.get("threshold_2", PARAMETER_CONFIG["threshold_2"]["default"]))
    processor.hough_dp = float(cfg.get("hough_dp", PARAMETER_CONFIG["hough_dp"]["default"]))
    processor.hough_mindist = int(cfg.get("hough_mindist", PARAMETER_CONFIG["hough_mindist"]["default"]))
    processor.hough_param1 = int(cfg.get("hough_param1", PARAMETER_CONFIG["hough_param1"]["default"]))
    processor.hough_param2 = int(cfg.get("hough_param2", PARAMETER_CONFIG["hough_param2"]["default"]))
    processor.hough_minr = int(cfg.get("hough_minr", PARAMETER_CONFIG["hough_minr"]["default"]))
    processor.hough_maxr = int(cfg.get("hough_maxr", PARAMETER_CONFIG["hough_maxr"]["default"]))

    renderer.canvas_width = int(cfg.get("canvas_width", 320))
    renderer.canvas_height = int(cfg.get("canvas_height", 280))

    processor.apply()
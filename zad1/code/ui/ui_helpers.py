import dearpygui.dearpygui as dpg
from zad1.code.ui.ui_parameters import SECTION_SPACING, FIELD_SPACING

def add_spacing(height): dpg.add_spacer(height=height)

def add_slider(label, tag, small_font, callback=None, default=280, min_v=1, max_v=900):
    is_float = any(isinstance(v, float) for v in (default, min_v, max_v))

    slider_func = dpg.add_slider_float if is_float else dpg.add_slider_int

    slider = slider_func(
        label=label,
        tag=tag,
        default_value=default,
        min_value=min_v,
        max_value=max_v,
        callback=callback,
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
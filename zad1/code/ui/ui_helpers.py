import dearpygui.dearpygui as dpg

FIELD_SPACING = 1
SECTION_SPACING = 4

def add_spacing(height): dpg.add_spacer(height=height)

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
import dearpygui.dearpygui as dpg

def update_texture(state, image_data, width, height):
    if image_data is None:
        return None

    with dpg.texture_registry():
        tex_id = dpg.add_dynamic_texture(width=width, height=height, default_value=image_data)

    # Delete previous texture if exists
    if state.texture_id is not None and dpg.does_item_exist(state.texture_id):
        dpg.delete_item(state.texture_id)

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
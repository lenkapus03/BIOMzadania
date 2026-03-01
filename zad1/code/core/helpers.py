import dearpygui.dearpygui as dpg

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
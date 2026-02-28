from zad1.code.commands.base_command import Command
import dearpygui.dearpygui as dpg

class UpdateCanvasCommand(Command):

    def __init__(self, image_manager, state):
        self.image_manager = image_manager
        self.state = state

    def execute(self, sender=None, app_data=None, user_data=None):

        # Ensure the sliders exist (they may not exist during initial setup)
        if not dpg.does_item_exist("canvas_width") or not dpg.does_item_exist("canvas_height"):
            return

        # Get current canvas dimensions from sliders
        w = dpg.get_value("canvas_width")
        h = dpg.get_value("canvas_height")

        # Generate scaled RGBA texture data from the processed image
        new_data = self.image_manager.get_scaled_texture(w, h)
        if new_data is None:
            return  # If image not loaded, do nothing

        # Create a new dynamic texture in the DPG texture registry
        with dpg.texture_registry():
            new_texture_id = dpg.add_dynamic_texture(width=w, height=h, default_value=new_data)

        # Update the image widget to use the new texture
        if dpg.does_item_exist("displayed_image_widget"):
            dpg.configure_item(
                "displayed_image_widget",
                texture_tag=new_texture_id,
                width=w,
                height=h
            )

        # Delete the previous texture to free GPU memory
        if self.state.texture_id is not None and dpg.does_item_exist(self.state.texture_id):
            dpg.delete_item(self.state.texture_id)

        # Store the current texture ID in state for future updates
        self.state.texture_id = new_texture_id
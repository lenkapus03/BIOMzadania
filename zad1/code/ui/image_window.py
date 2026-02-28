import dearpygui.dearpygui as dpg

class ImageWindow:
    def __init__(self, state, width, height, pos_x):
        self.state = state
        self.width = width
        self.height = height
        self.pos_x = pos_x

    def build(self):
        with dpg.window(
            label="View image",
            width=self.width,
            height=self.height,
            pos=(self.pos_x, 0),
        ):
            self._create_image_widget()

    # Image logic
    def _create_image_widget(self):
        if self.state.texture_id is None:
            return

        dpg.add_image(
            self.state.texture_id,
            tag="displayed_image_widget",
        )

    def update_texture(self, texture_id, width=None, height=None):
        if not dpg.does_item_exist("displayed_image_widget"):
            return

        dpg.configure_item(
            "displayed_image_widget",
            texture_tag=texture_id,
            width=width,
            height=height,
        )
import cv2
import numpy as np

class Renderer:
    def __init__(self, processor, state=None, canvas_width=320, canvas_height=280):
        self.texture_id = None
        self.processor = processor
        self.state = state
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.preview_original = False

    def _build_canvas(self, img):
        img_h, img_w = img.shape[:2]
        canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

        x_offset = (self.canvas_width - img_w) // 2
        y_offset = (self.canvas_height - img_h) // 2

        img_x1 = max(0, -x_offset)
        img_y1 = max(0, -y_offset)
        canv_x1 = max(0, x_offset)
        canv_y1 = max(0, y_offset)
        draw_w = min(img_w - img_x1, self.canvas_width - canv_x1)
        draw_h = min(img_h - img_y1, self.canvas_height - canv_y1)

        if draw_w > 0 and draw_h > 0:
            canvas[canv_y1:canv_y1 + draw_h, canv_x1:canv_x1 + draw_w] = \
                img[img_y1:img_y1 + draw_h, img_x1:img_x1 + draw_w]
        return canvas

    def get_scaled_texture(self):
        if self.processor.processed_image is None:
            return None

        processed_canvas = self._build_canvas(self.processor.processed_image)

        if self.processor.preview_original:
            display_canvas = self._build_canvas(self.processor.original_image)
        else:
            display_canvas = processed_canvas

        if self.processor.show_hough:
            original_shape = self.processor.original_image.shape if self.processor.original_image is not None else None
            display_canvas = self.processor.hough(
                display_canvas,
                self.state.active_circle,
                original_shape,
                show_rejected=self.processor.show_rejected_circles,
                detect_on=processed_canvas
            )

        canvas = cv2.cvtColor(display_canvas, cv2.COLOR_BGR2RGBA)
        return canvas.astype(np.float32).flatten() / 255.0

    def refresh_texture(self, apply_processor=True):
        if apply_processor:
            self.processor.apply()

        tex_data = self.get_scaled_texture()
        if tex_data is None:
            return None

        # Use helper to update texture in DPG
        from zad1.code.core.helpers import update_texture
        self.texture_id = update_texture(self, tex_data, self.canvas_width, self.canvas_height)
        return self.texture_id
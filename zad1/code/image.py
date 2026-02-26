import cv2
import numpy as np

class ImageManager:
    def __init__(self):
        self.original_image = None
        self.processed_image = None

    def load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image from {path}")
        self.original_image = img
        self.processed_image = img.copy()

    def reset(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()

    def get_scaled_texture(self, canvas_width, canvas_height):
        if self.processed_image is None:
            return None

        img = self.processed_image
        img_h, img_w = img.shape[:2]

        # create a white canvas of the NEW requested size
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # calculate centering
        x_offset = (canvas_width - img_w) // 2
        y_offset = (canvas_height - img_h) // 2

        # determine slice coordinates (handling cases where canvas < image)
        img_x1 = max(0, -x_offset)
        img_y1 = max(0, -y_offset)
        canv_x1 = max(0, x_offset)
        canv_y1 = max(0, y_offset)

        draw_w = min(img_w - img_x1, canvas_width - canv_x1)
        draw_h = min(img_h - img_y1, canvas_height - canv_y1)

        if draw_w > 0 and draw_h > 0:
            canvas[canv_y1:canv_y1 + draw_h, canv_x1:canv_x1 + draw_w] = \
                img[img_y1:img_y1 + draw_h, img_x1:img_x1 + draw_w]

        # convert to RGBA float format for Dear PyGui
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA)
        return canvas.astype(np.float32).flatten() / 255.0
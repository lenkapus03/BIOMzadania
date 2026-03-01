import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, original_image=None):
        self.original_image = original_image
        self.processed_image = original_image.copy() if original_image is not None else None

        # Settings
        self.use_histogram_eq = False
        self.canvas_width = 320
        self.canvas_height = 280

        self.use_clahe = False
        self.clip_limit = 2
        self.grid_size = 8

    # Source: https://www.freedomvc.com/index.php/2021/09/11/color-image-histograms/
    def histogram_equalization(self, image):
        # Split into B, G, R channels
        b, g, r = cv2.split(image)

        # Equalize each channel separately
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)

        # Merge equalized channels back into a single BGR image
        equalized = cv2.merge((b_eq, g_eq, r_eq))
        return equalized

    # Source: https://medium.com/@lin.yong.hui.jason/histogram-equalization-for-color-images-using-opencv-655ae13b9dd0
    def clahe(self, image):

        # Create CLAHE object with given parameters
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.grid_size, self.grid_size)
        )

        # Apply CLAHE to each channel separately
        blue_clahe = clahe.apply(image[:, :, 0])
        green_clahe = clahe.apply(image[:, :, 1])
        red_clahe = clahe.apply(image[:, :, 2])

        # Merge channels back into BGR image
        return cv2.merge((blue_clahe, green_clahe, red_clahe))

    def apply(self):
        if self.original_image is None:
            return None

        img = self.original_image.copy()

        if self.use_histogram_eq:
            img = self.histogram_equalization(img)

        if self.use_clahe:
            img = self.clahe(img)
        # ... other transformations go here

        self.processed_image = img
        return self.processed_image

    def get_scaled_texture(self):
        # If there is no processed image, return None (nothing to display)
        if self.processed_image is None:
            return None

        # Get height and width of the processed image
        img_h, img_w = self.processed_image.shape[:2]

        # Create a blank canvas of the desired size, filled with white (255)
        # Canvas shape: (canvas_height, canvas_width, 3) for RGB
        canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

        # Compute offsets to center the image on the canvas
        x_offset = (self.canvas_width - img_w) // 2
        y_offset = (self.canvas_height - img_h) // 2

        # Calculate the start coordinates in the image and canvas for drawing
        # Handles cases where the image is larger than the canvas (cropping)
        img_x1 = max(0, -x_offset)  # start x in image
        img_y1 = max(0, -y_offset)  # start y in image
        canv_x1 = max(0, x_offset)  # start x in canvas
        canv_y1 = max(0, y_offset)  # start y in canvas

        # Determine how much width/height to draw (handles partial overlap)
        draw_w = min(img_w - img_x1, self.canvas_width - canv_x1)
        draw_h = min(img_h - img_y1, self.canvas_height - canv_y1)

        # If there is a valid area to draw, copy the image onto the canvas
        if draw_w > 0 and draw_h > 0:
            canvas[canv_y1:canv_y1 + draw_h, canv_x1:canv_x1 + draw_w] = \
                self.processed_image[img_y1:img_y1 + draw_h, img_x1:img_x1 + draw_w]

        # Convert from BGR (OpenCV default) to RGBA (needed by DearPyGui)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA)

        # Flatten the image to a 1D float array with values in range [0, 1]
        # This is the format required by dpg.add_dynamic_texture
        return canvas.astype(np.float32).flatten() / 255.0
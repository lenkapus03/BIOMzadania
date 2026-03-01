import cv2

class ImageProcessor:
    def __init__(self, original_image=None):
        self.original_image = original_image
        self.processed_image = original_image.copy() if original_image is not None else None

        # Settings
        self.use_histogram_eq = False

        self.use_clahe = False
        self.clip_limit = 2
        self.grid_size = 8

    # Source: https://www.freedomvc.com/index.php/2021/09/11/color-image-histograms/
    def histogram_equalization(self, image):
        b, g, r = cv2.split(image)
        return cv2.merge([
            cv2.equalizeHist(b),
            cv2.equalizeHist(g),
            cv2.equalizeHist(r)
        ])

    # Source: https://medium.com/@lin.yong.hui.jason/histogram-equalization-for-color-images-using-opencv-655ae13b9dd0
    def clahe(self, image):
        clahe_obj = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.grid_size, self.grid_size)
        )
        return cv2.merge([
            clahe_obj.apply(image[:, :, 0]),
            clahe_obj.apply(image[:, :, 1]),
            clahe_obj.apply(image[:, :, 2])
        ])

    def apply(self):
        if self.original_image is None:
            self.processed_image = None
            return None

        img = self.original_image.copy()
        if self.use_histogram_eq:
            img = self.histogram_equalization(img)
        if self.use_clahe:
            img = self.clahe(img)

        self.processed_image = img
        return self.processed_image
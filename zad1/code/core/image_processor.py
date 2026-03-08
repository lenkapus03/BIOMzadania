import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, original_image=None):
        self.original_image = original_image
        self.processed_image = original_image.copy() if original_image is not None else None

        # Settings
        self.use_histogram_eq = False

        self.use_clahe = False
        self.clip_limit = 2
        self.grid_size = 8

        self.use_blur = False
        self.gauss_kernel = 5
        self.gauss_sigma = 1.0

        self.use_canny = False
        self.canny_threshold_1 = 50
        self.canny_threshold_2 = 150

        self.show_hough = False
        self.hough_dp = 1.2
        self.hough_mindist = 50
        self.hough_param1 = 100  # higher canny threshold (internal)
        self.hough_param2 = 30  # accumulator threshold
        self.hough_minr = 0
        self.hough_maxr = 0

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

    # Source: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    def gaussian_blur(self, image):
        if self.gauss_kernel == 0 and self.gauss_sigma == 0:
            return image
        kernel = self.gauss_kernel
        if kernel > 0 and kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(image, (kernel, kernel), self.gauss_sigma)

    # Source: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    def canny(self, image):
        edges = cv2.Canny(image, self.canny_threshold_1, self.canny_threshold_2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def hough(self, image):
        # Convert image to greyscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            gray_image, # Vstupné pole
            cv2.HOUGH_GRADIENT, # Metóda
            dp=self.hough_dp,  # Rozlíšenie akumulátora (1=rovnaké as input, 2=polovica rozlíšenia)
            minDist=self.hough_mindist,  # min. vzdialenosť stredov kružníc
            param1=self.hough_param1,  # vyššia hranica v Cannyho detektore
            param2=self.hough_param2,  # hranica akumulátora pre stredy kručníc
            minRadius=self.hough_minr,  # min. veľkosť kružníc
            maxRadius=self.hough_maxr  # max. veľkosť kružníc
        )

        # Zobrazenie detekovaných kruhov
        result = image.copy()
        if circles is not None:
            for c in circles[0, :]:
                center = (int(c[0]), int(c[1]))
                radius = int(c[2])
                cv2.circle(result, center, 1, (0, 100, 100), 3)  # stred
                cv2.circle(result, center, radius, (255, 0, 255), 3)  # kružnica
        return result

    def apply(self):
        if self.original_image is None:
            self.processed_image = None
            return None

        img = self.original_image.copy()
        if self.use_histogram_eq:
            img = self.histogram_equalization(img)
        if self.use_clahe:
            img = self.clahe(img)
        if self.use_blur and not self.use_canny:
            img = self.gaussian_blur(img)
        if self.use_canny:
            img = self.canny(img)

        self.processed_image = img
        return self.processed_image
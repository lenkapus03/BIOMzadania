import cv2
import numpy as np

CIRCLE_COLORS = {
    "iris":      (255, 0, 0),    # modrá
    "pupil":     (0, 255, 0),    # zelená
    "upper_lid": (0, 165, 255),  # oranžová
    "lower_lid": (0, 0, 255),    # červená
}

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
        self.show_rejected_circles = False
        self.hough_dp = 1.2
        self.hough_mindist = 50
        self.hough_param1 = 100  # higher canny threshold (internal)
        self.hough_param2 = 30  # accumulator threshold
        self.hough_minr = 0
        self.hough_maxr = 0

        self.preview_original = False

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

    def hough(self, image, active_circle="iris", original_shape=None, show_rejected=True, detect_on=None):
        # Detect circles on detect_on if provided, otherwise on image
        source = detect_on if detect_on is not None else image

        # Convert image to greyscale
        gray_image = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

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

        result = image.copy()
        color = CIRCLE_COLORS.get(active_circle, (255, 0, 255))

        if circles is not None:
            filter_shape = original_shape if original_shape is not None else image.shape
            accepted = self.filter_circles(circles, filter_shape, active_circle, canvas_shape=image.shape)
            accepted_set = set()
            if accepted is not None:
                for c in accepted[0, :]:
                    accepted_set.add((int(c[0]), int(c[1]), int(c[2])))

            for c in circles[0, :]:
                center = (int(c[0]), int(c[1]))
                radius = int(c[2])
                is_accepted = (center[0], center[1], radius) in accepted_set

                if not is_accepted and not show_rejected:
                    continue

                draw_color = color if is_accepted else (150, 150, 150)
                cv2.circle(result, center, 1, draw_color, 1)
                cv2.circle(result, center, radius, draw_color, 1)

        return result

    def filter_circles(self, circles, image_shape, active_circle, canvas_shape=None):
        if circles is None:
            return None

        h, w = image_shape[:2]

        if canvas_shape is not None:
            ch, cw = canvas_shape[:2]
            # Stred canvasu = stred obrazka v canvas súradniciach (obrazok je vycentrovaný)
            cx, cy = cw // 2, ch // 2
            # Hranice pôvodného obrazka v canvas súradniciach
            img_x1 = cx - w // 2
            img_y1 = cy - h // 2
            img_x2 = img_x1 + w
            img_y2 = img_y1 + h
        else:
            cx, cy = w // 2, h // 2
            img_x1, img_y1, img_x2, img_y2 = 0, 0, w, h

        best = None
        best_score = float('inf')

        for c in circles[0, :]:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            # Vzdialenosť stredu kruhu od stredu obrazka — čím menšia, tým lepšie
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

            if active_circle == "iris":
                # Dúhovka — stredne veľký kruh (20%-60% šírky obrazka)
                if not (w * 0.2 < r < w * 0.6):
                    continue
                # Celý kruh sa musí zmestiť do pôvodného obrazka
                if x - r < img_x1 or x + r > img_x2 or y - r < img_y1 or y + r > img_y2:
                    continue
                # Preferujeme kruh najbližší k stredu obrazka
                score = dist

            elif active_circle == "pupil":
                # Zrenička — malý kruh (menej ako 25% šírky obrazka)
                if not (r < w * 0.25):
                    continue
                # Celý kruh sa musí zmestiť do pôvodného obrazka
                if x - r < img_x1 or x + r > img_x2 or y - r < img_y1 or y + r > img_y2:
                    continue
                # Preferujeme kruh najbližší k stredu obrazka
                score = dist

            elif active_circle == "upper_lid":
                # Horné viečko — veľký polomer (viac ako 40% šírky obrazka)
                if not (r > w * 0.4):
                    continue
                # Stred kruhu musí byť v DOLNEJ polovici canvasu
                # (kruh s veľkým polomerom ktorého stred je dole pokrýva hornú časť obrazka)
                if y < cy:
                    continue
                # Horizontálne musí byť stred blízko stredu obrazka
                if abs(x - cx) > w * 0.4:
                    continue
                # Preferujeme kruh najbližší k stredu obrazka
                score = dist

            elif active_circle == "lower_lid":
                # Dolné viečko — veľký polomer (viac ako 40% šírky obrazka)
                if not (r > w * 0.4):
                    continue
                # Stred kruhu musí byť v HORNEJ polovici canvasu
                # (kruh s veľkým polomerom ktorého stred je hore pokrýva dolnú časť obrazka)
                if y > cy:
                    continue
                # Horizontálne musí byť stred blízko stredu obrazka
                if abs(x - cx) > w * 0.4:
                    continue
                # Preferujeme kruh najbližší k stredu obrazka
                score = dist

            else:
                continue

            if score < best_score:
                best_score = score
                best = c

        if best is None:
            return None
        return np.array([[best]], dtype=np.float32)

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
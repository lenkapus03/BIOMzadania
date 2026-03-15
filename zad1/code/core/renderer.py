import cv2
import numpy as np
from zad1.code.core.helpers import apply_params

class Renderer:
    def __init__(self, processor, state=None, canvas_width=320, canvas_height=280):
        self.texture_id = None
        self.processor = processor
        self.state = state
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.preview_original = False
        self.show_ground_truth = False
        self.show_all_circles = False

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
            display_canvas = processed_canvas.copy()

        original_shape = self.processor.original_image.shape if self.processor.original_image is not None else None

        # Rozmery pre výslednu textúru — môžu sa zmeniť pri show_all_circles
        display_w = self.canvas_width
        display_h = self.canvas_height

        # Detekuj a nakresli každú kružnicu s jej vlastnými parametrami
        if self.show_all_circles:
            # Nájdi maximálne rozlíšenie cez všetky kružnice pre zobrazenie
            max_w = int(max(
                (cfg.get("canvas_width", self.canvas_width) for cfg in self.state.circle_params.values() if cfg),
                default=self.canvas_width
            ))
            max_h = int(max(
                (cfg.get("canvas_height", self.canvas_height) for cfg in self.state.circle_params.values() if cfg),
                default=self.canvas_height
            ))

            backup_w = self.canvas_width
            backup_h = self.canvas_height
            backup_processed = self.processor.processed_image.copy()
            active_cfg = self.state.circle_params.get(self.state.active_circle, {})

            # Nastav maximálne rozlíšenie pre display canvas
            self.canvas_width = max_w
            self.canvas_height = max_h
            display_w = max_w
            display_h = max_h

            # Display canvas vždy z pôvodného obrazka — nechceme zobrazovať spracovaný obraz
            display_canvas = self._build_canvas(self.processor.original_image)

            for circle_name in ["iris", "pupil", "upper_lid", "lower_lid"]:
                cfg = self.state.circle_params.get(circle_name, {})
                if not cfg:
                    continue

                # Nastav parametre tejto kružnice vrátane jej canvas rozlíšenia
                apply_params(cfg, self.processor, self)

                # Postav detect canvas s rozlíšením tejto kružnice
                circle_processed = self._build_canvas(self.processor.processed_image)
                circle_canvas_shape = circle_processed.shape

                # Vypočítaj offset medzi detect canvas a display canvas
                # Keďže obrazok je vycentrovaný, menší canvas má iný offset ako max canvas
                circle_w = int(cfg.get("canvas_width", backup_w))
                circle_h = int(cfg.get("canvas_height", backup_h))
                detect_offset_x = (max_w - circle_w) // 2
                detect_offset_y = (max_h - circle_h) // 2

                # Obnov max rozmery pre kreslenie na display canvas
                self.canvas_width = max_w
                self.canvas_height = max_h

                display_canvas = self.processor.hough(
                    display_canvas,
                    circle_name,
                    original_shape,
                    show_rejected=False,
                    detect_on=circle_processed,
                    filter_canvas_shape=circle_canvas_shape,
                    draw_offset=(detect_offset_x, detect_offset_y)
                )

            # Obnov pôvodné parametre a rozmery
            self.canvas_width = backup_w
            self.canvas_height = backup_h
            self.processor.processed_image = backup_processed
            if active_cfg:
                apply_params(active_cfg, self.processor, self)
                self.canvas_width = backup_w
                self.canvas_height = backup_h

        elif self.processor.show_hough:
            display_canvas = self.processor.hough(
                display_canvas,
                self.state.active_circle,
                original_shape,
                show_rejected=self.processor.show_rejected_circles,
                detect_on=processed_canvas
            )

        # Nakresli ground truth kružnice
        if self.show_ground_truth and self.state.ground_truth_circles is not None:
            circles_to_draw = (
                self.state.ground_truth_circles  # všetky pri show_all_circles
                if self.show_all_circles
                else {self.state.active_circle: self.state.ground_truth_circles.get(self.state.active_circle)}
            )
            display_canvas = self._draw_ground_truth(display_canvas, circles_to_draw)

        canvas = cv2.cvtColor(display_canvas, cv2.COLOR_BGR2RGBA)
        return canvas.astype(np.float32).flatten() / 255.0, display_w, display_h

    def refresh_texture(self, apply_processor=True):
        if apply_processor:
            self.processor.apply()

        result = self.get_scaled_texture()
        if result is None:
            return None

        # Rozbaľ textúru a rozmery — rozmery sa môžu líšiť pri show_all_circles
        tex_data, display_w, display_h = result

        from zad1.code.core.helpers import update_texture
        self.texture_id = update_texture(self, tex_data, display_w, display_h)
        return self.texture_id

    def _draw_ground_truth(self, canvas, circles):
        result = canvas.copy()
        img_h, img_w = self.processor.original_image.shape[:2]
        x_offset = (self.canvas_width - img_w) // 2
        y_offset = (self.canvas_height - img_h) // 2

        for circle_name, gt in circles.items():
            if gt is None:
                continue
            gx, gy, gr = gt
            print(f"[GT] {circle_name}: gx={gx}, gy={gy}, gr={gr}, canvas_cy={gy + y_offset}")
            if gr <= 0:
                continue
            cx = gx + x_offset
            cy = gy + y_offset
            cv2.circle(result, (int(cx), int(cy)), int(gr), (255, 0, 255), 1)  # fialová
            cv2.circle(result, (int(cx), int(cy)), 2, (255, 0, 255), 2)
        return result
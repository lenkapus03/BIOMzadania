# import os
# import json
# import cv2
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
#
# # ──────────────────────────────────────────────
# # Konfigurácia ciest
# # ──────────────────────────────────────────────
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# CONFIG_FOLDER = CURRENT_DIR
# DATA_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "data"))
# CSV_PATH = os.path.join(DATA_FOLDER, "iris_annotation.csv")
#
# # Mapovanie kružníc na stĺpce v CSV
# # Poradie: duhovka, zrenicka, dolne viecko, horne viecko
# CIRCLE_NAMES = ["iris", "pupil", "lower_lid", "upper_lid"]
# CIRCLE_CSV_INDEX = {
#     "iris":      1,
#     "pupil":     2,
#     "lower_lid": 3,
#     "upper_lid": 4,
# }
#
# IOU_THRESHOLD = 0.75
#
#
# # ──────────────────────────────────────────────
# # Načítanie dát
# # ──────────────────────────────────────────────
# @dataclass
# class ImageRecord:
#     image: str
#     center_x_1: int
#     center_y_1: int
#     polomer_1: int
#     center_x_2: int
#     center_y_2: int
#     polomer_2: int
#     center_x_3: int
#     center_y_3: int
#     polomer_3: int
#     center_x_4: int
#     center_y_4: int
#     polomer_4: int
#
#
# def load_records(csv_path, n=100):
#     df = pd.read_csv(csv_path)
#     n = min(n, len(df))
#     sampled_df = df.sample(n=n, random_state=42)
#     return [ImageRecord(**row.to_dict()) for _, row in sampled_df.iterrows()]
#
#
# def load_config(circle_name):
#     path = os.path.join(CONFIG_FOLDER, f"{circle_name}_config.json")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Config not found: {path}")
#     with open(path) as f:
#         return json.load(f)
#
#
# # ──────────────────────────────────────────────
# # IoU pre kruhy
# # Source: https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
# # ──────────────────────────────────────────────
# def circle_iou(cx1, cy1, r1, cx2, cy2, r2):
#     d = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
#
#     # Žiadny prienik
#     if d >= r1 + r2:
#         return 0.0
#
#     # Jeden kruh je vnútri druhého
#     if d <= abs(r1 - r2):
#         smaller = min(r1, r2)
#         larger = max(r1, r2)
#         intersection = np.pi * smaller ** 2
#         union = np.pi * larger ** 2
#         return intersection / union
#
#     # Čiastočný prienik
#     r1_sq, r2_sq, d_sq = r1 ** 2, r2 ** 2, d ** 2
#     alpha = np.arccos((d_sq + r1_sq - r2_sq) / (2 * d * r1))
#     beta = np.arccos((d_sq + r2_sq - r1_sq) / (2 * d * r2))
#     intersection = r1_sq * alpha + r2_sq * beta - r1_sq * np.sin(2 * alpha) / 2 - r2_sq * np.sin(2 * beta) / 2
#     union = np.pi * (r1_sq + r2_sq) - intersection
#     return intersection / union
#
#
# # ──────────────────────────────────────────────
# # Spracovanie obrazka a detekcia kruhu
# # Source: https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
# # ──────────────────────────────────────────────
# def detect_circle(canvas, cfg, circle_name, original_shape):
#     # canvas uz obsahuje spracovany obrazok s paddingom
#     gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
#
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         dp=float(cfg["hough_dp"]),
#         minDist=int(cfg["hough_mindist"]),
#         param1=int(cfg["hough_param1"]),
#         param2=int(cfg["hough_param2"]),
#         minRadius=int(cfg["hough_minr"]),
#         maxRadius=int(cfg["hough_maxr"])
#     )
#
#     if circles is None:
#         return None
#
#     return filter_best_circle(circles, original_shape, canvas.shape, circle_name)
#
#
# def filter_best_circle(circles, image_shape, canvas_shape, active_circle):
#     h, w = image_shape[:2]
#     ch, cw = canvas_shape[:2]
#     cx, cy = cw // 2, ch // 2
#     img_x1 = cx - w // 2
#     img_y1 = cy - h // 2
#     img_x2 = img_x1 + w
#     img_y2 = img_y1 + h
#
#     best = None
#     best_score = float('inf')
#
#     for c in circles[0, :]:
#         x, y, r = int(c[0]), int(c[1]), int(c[2])
#         dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
#
#         if active_circle == "iris":
#             if not (w * 0.2 < r < w * 0.6):
#                 continue
#             if x - r < img_x1 or x + r > img_x2 or y - r < img_y1 or y + r > img_y2:
#                 continue
#
#         elif active_circle == "pupil":
#             if not (r < w * 0.25):
#                 continue
#             if x - r < img_x1 or x + r > img_x2 or y - r < img_y1 or y + r > img_y2:
#                 continue
#
#         elif active_circle == "upper_lid":
#             if not (r > w * 0.4):
#                 continue
#             if y < cy:
#                 continue
#             if abs(x - cx) > w * 0.4:
#                 continue
#             if (y - r) > cy:
#                 continue
#
#         elif active_circle == "lower_lid":
#             if not (r > w * 0.4):
#                 continue
#             if y > cy:
#                 continue
#             if abs(x - cx) > w * 0.4:
#                 continue
#             if (y + r) < cy:
#                 continue
#
#         if dist < best_score:
#             best_score = dist
#             best = c
#
#     return best
#
#
# def evaluate():
#     records = load_records(CSV_PATH, n=100)
#
#     results = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CIRCLE_NAMES}
#
#     for record in records:
#         img_path = os.path.normpath(os.path.join(DATA_FOLDER, record.image.replace("\\", "/")))
#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"Cannot load image: {img_path}")
#             continue
#
#         original_shape = image.shape
#
#         for circle_name in CIRCLE_NAMES:
#             idx = CIRCLE_CSV_INDEX[circle_name]
#             gt_x = getattr(record, f"center_x_{idx}")
#             gt_y = getattr(record, f"center_y_{idx}")
#             gt_r = getattr(record, f"polomer_{idx}")
#
#             # Preskočiť neplatné anotácie
#             if gt_r <= 0:
#                 continue
#
#             try:
#                 cfg = load_config(circle_name)
#             except FileNotFoundError as e:
#                 print(e)
#                 continue
#
#             # Canvas rozmery z konfigurácie
#             canvas_w = int(cfg.get("canvas_width", image.shape[1]))
#             canvas_h = int(cfg.get("canvas_height", image.shape[0]))
#
#             # Spracuj obrazok podľa konfigurácie
#             img = image.copy()
#             if cfg.get("use_histogram_eq"):
#                 b, g, r = cv2.split(img)
#                 img = cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])
#             if cfg.get("use_clahe"):
#                 clahe_obj = cv2.createCLAHE(
#                     clipLimit=float(cfg["clahe_clip"]),
#                     tileGridSize=(int(cfg["clahe_tile"]), int(cfg["clahe_tile"]))
#                 )
#                 img = cv2.merge([
#                     clahe_obj.apply(img[:, :, 0]),
#                     clahe_obj.apply(img[:, :, 1]),
#                     clahe_obj.apply(img[:, :, 2])
#                 ])
#             if cfg.get("use_blur") and not cfg.get("use_canny"):
#                 kernel = int(cfg["gauss_kernel"])
#                 if kernel > 0:
#                     if kernel % 2 == 0:
#                         kernel += 1
#                     img = cv2.GaussianBlur(img, (kernel, kernel), float(cfg["gauss_sigma"]))
#             if cfg.get("use_canny"):
#                 edges = cv2.Canny(img, int(cfg["threshold_1"]), int(cfg["threshold_2"]))
#                 img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#
#             # Vytvor canvas zo spracovaného obrazka s paddingom
#             canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
#             img_h, img_w = img.shape[:2]
#             x_offset = (canvas_w - img_w) // 2
#             y_offset = (canvas_h - img_h) // 2
#             canv_x1 = max(0, x_offset)
#             canv_y1 = max(0, y_offset)
#             img_x1 = max(0, -x_offset)
#             img_y1 = max(0, -y_offset)
#             draw_w = min(img_w - img_x1, canvas_w - canv_x1)
#             draw_h = min(img_h - img_y1, canvas_h - canv_y1)
#             if draw_w > 0 and draw_h > 0:
#                 canvas[canv_y1:canv_y1 + draw_h, canv_x1:canv_x1 + draw_w] = \
#                     img[img_y1:img_y1 + draw_h, img_x1:img_x1 + draw_w]
#
#             # Detekuj kruh na canvase so spracovaným obrazkom
#             detected = detect_circle(canvas, cfg, circle_name, original_shape)
#
#             if detected is None:
#                 results[circle_name]["fn"] += 1
#                 continue
#
#             det_x, det_y, det_r = int(detected[0]), int(detected[1]), int(detected[2])
#
#             # Prepočet súradníc z canvas priestoru do obrazkového priestoru
#             det_x_img = det_x - canv_x1 + img_x1
#             det_y_img = det_y - canv_y1 + img_y1
#
#             iou = circle_iou(det_x_img, det_y_img, det_r, gt_x, gt_y, gt_r)
#
#             if iou >= IOU_THRESHOLD:
#                 results[circle_name]["tp"] += 1
#             else:
#                 results[circle_name]["fp"] += 1
#                 results[circle_name]["fn"] += 1
#
#     # Výpis výsledkov
#     print(f"\n{'Kružnica':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
#     print("-" * 60)
#     for name in CIRCLE_NAMES:
#         tp = results[name]["tp"]
#         fp = results[name]["fp"]
#         fn = results[name]["fn"]
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
#         print(f"{name:<12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {tp:>6} {fp:>6} {fn:>6}")
#
#
# if __name__ == "__main__":
#     evaluate()
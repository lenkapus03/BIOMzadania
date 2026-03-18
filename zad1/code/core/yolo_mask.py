import cv2

# Source: https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format
def mask_to_yolo_polygon(mask, img_w, img_h, class_id=0):
    """Konvertuje binárnu masku na YOLO polygon formát."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue  # nie je polygon

        normalized_points = []
        for point in contour.squeeze():
            x = point[0] / img_w
            y = point[1] / img_h
            # zabezpeč rozsah 0-1
            if 0 <= x <= 1 and 0 <= y <= 1:
                normalized_points.append(f"{x:.6f} {y:.6f}")

        if normalized_points:
            lines.append(f"{class_id} " + " ".join(normalized_points))

    return "\n".join(lines) if lines else None
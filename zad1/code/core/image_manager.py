import cv2

class ImageManager:
    def __init__(self):
        self.original_image = None
        self.processed_image = None

    def load_image(self, path: str):
        """Načíta obraz zo súboru. Vyhodí ValueError ak súbor neexistuje alebo nie je čitateľný."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image from {path}")
        self.original_image = img
        self.processed_image = img.copy()

    def reset(self):
        """Obnoví spracovaný obraz na pôvodný."""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()

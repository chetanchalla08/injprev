"""Optional pretrained object detector (YOLOX-S, Apache-2.0, COCO classes) run on CPU through OpenCV.

Used only for the ball / racket evaluation and, if enabled, as extra evidence for racket
position. COCO has "sports ball" and "tennis racket" classes, but the model was not trained
for small, fast, motion-blurred tennis balls; docs/VALIDATION.md records how it compared
with the motion-based ball finder on the evaluation set.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PERSON, SPORTS_BALL, TENNIS_RACKET = 0, 32, 38
SIZE = 640


class YoloxDetector:
    def __init__(self, model_path: Path) -> None:
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        grids, strides = [], []
        for s in (8, 16, 32):
            n = SIZE // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack([xv, yv], 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def detect(self, frame_bgr: np.ndarray, classes: tuple[int, ...] = (SPORTS_BALL, TENNIS_RACKET), min_score: float = 0.25) -> list[tuple[int, float, float, float, float, float]]:
        """Returns [(class, score, x0, y0, x1, y1)] in pixels of the given image."""
        h, w = frame_bgr.shape[:2]
        r = min(SIZE / h, SIZE / w)
        canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
        canvas[: int(h * r), : int(w * r)] = cv2.resize(frame_bgr, (int(w * r), int(h * r)))
        self.net.setInput(canvas.transpose(2, 0, 1)[None].astype(np.float32))
        o = self.net.forward()[0]
        xy = (o[:, :2] + self.grid) * self.stride
        wh = np.exp(np.clip(o[:, 2:4], -10, 10)) * self.stride
        out = []
        for c in classes:
            sc = o[:, 4] * o[:, 5 + c]
            keep = np.where(sc >= min_score)[0]
            if not keep.size:
                continue
            boxes = np.concatenate([xy[keep] - wh[keep] / 2, wh[keep]], 1) / r
            for k in cv2.dnn.NMSBoxes(boxes.tolist(), sc[keep].tolist(), min_score, 0.45):
                x, y, bw, bh = boxes[int(k)]
                out.append((c, float(sc[keep][int(k)]), float(x), float(y), float(x + bw), float(y + bh)))
        return out

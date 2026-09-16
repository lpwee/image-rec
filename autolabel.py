#!/usr/bin/env python3
"""Auto-label SC2079 symbol-card images with YOLO bounding boxes.

Detection is classical CV. The card is a square-ish bright region with a
high-contrast symbol inside, sitting on the dark face of the obstacle. Candidate
blobs come from two sources: global thresholds (card brighter than everything),
and "box first" (threshold locally inside each large dark blob, for dim scenes
where the card is only bright relative to the obstacle). Candidates are then
filtered on shape and scored on how well the left/right/bottom surround is dark
and whether the symbol sits centered inside. There is no fallback box: an image
with no plausible card is reported as failed and left out of the dataset.

Class indices are fixed as ``image_id - 11`` (IDs 11-40 -> indices 0-29), so
labels stay consistent between photobooth sidecars and this converter.

Two ways to use it:

1. Imported by photobooth.py, which calls write_sidecar_label() after every
   capture to drop a YOLO .txt next to the .jpg.

2. Standalone converter: reads the ImageFolder dataset and emits an
   Ultralytics detection dataset (images/ + labels/ + data.yaml), preferring
   photobooth sidecar labels and running detection for images without one::

       python3 autolabel.py                  # dataset/ -> dataset_yolo/
       python3 autolabel.py --force          # redo existing labels
       python3 autolabel.py --no-review      # skip annotated review images

   Review images with the detected box drawn on are written to
   dataset_yolo/review/<split>/ - flip through them and hand-fix (or recapture)
   anything listed in the failure summary.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

CLASS_ID_BASE = 11  # image ID 11 -> class index 0
CLASS_DIR_RE = re.compile(r"^(\d+)_(.+)$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

PAD_FRAC = 0.03
MIN_AREA_FRAC, MAX_AREA_FRAC = 0.002, 0.85
MIN_ASPECT, MAX_ASPECT = 0.5, 2.0
MIN_SOLIDITY = 0.75
MIN_FILL = 0.6
MIN_CONTRAST = 50
THRESHOLDS = (120, 150, 180, 210)
SIDE_FRAC = 0.25
SIDE_DARK = 95
MIN_SIDE_COVER = 0.4
DARK_FACE = 70          # obstacle faces / the stop card are below this gray level
MIN_FACE_FRAC = 0.01    # dark blob must cover at least this much of the frame


def _side_score(blur, x, y, w, h):
    H, W = blur.shape
    bw, bh = max(2, int(w * SIDE_FRAC)), max(2, int(h * SIDE_FRAC))
    bands = [(x - bw, y, x, y + h), (x + w, y, x + w + bw, y + h), (x, y + h, x + w, y + h + bh)]
    dark = usable = 0
    for x0, y0, x1, y1 in bands:
        full = (x1 - x0) * (y1 - y0)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < MIN_SIDE_COVER * full:
            continue
        usable += 1
        dark += blur[cy0:cy1, cx0:cx1].mean() < SIDE_DARK
    return dark, usable


def _symbol_centered(roi_dark):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(roi_dark.astype(np.uint8), 8)
    if n <= 1:
        return False
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    return x > 0 and y > 0 and x + w < roi_dark.shape[1] and y + h < roi_dark.shape[0]


def _masks(blur_crop, edges_crop, thresholds):
    """Yield candidate bright masks for a (cropped) frame at several thresholds."""
    for t in thresholds:
        _, bright = cv2.threshold(blur_crop, t, 255, cv2.THRESH_BINARY)
        cut = bright.copy()
        cut[edges_crop > 0] = 0
        yield cv2.morphologyEx(cut, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        yield cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


class _Scorer:
    """Filters and scores candidate contours; keeps the best (score, (x, y, w, h))."""

    def __init__(self, blur):
        self.blur = blur
        self.H, self.W = blur.shape
        self.frame_area = self.W * self.H
        self.seen = set()
        self.best = None
        self.debug = None

    def consider(self, contour, ox=0, oy=0, min_contrast=MIN_CONTRAST, strict_sides=True, min_dark=0.04):
        blur, W, H = self.blur, self.W, self.H
        area = cv2.contourArea(contour)
        if not MIN_AREA_FRAC * self.frame_area < area < MAX_AREA_FRAC * self.frame_area:
            return
        x, y, w, h = cv2.boundingRect(contour)
        x, y = x + ox, y + oy
        if not MIN_ASPECT < w / h < MAX_ASPECT:
            return
        if x <= 2 and x + w >= W - 2:
            return
        key = (x // 6, y // 6, w // 6, h // 6)
        if key in self.seen:
            return
        self.seen.add(key)
        rect_area = np.prod(cv2.minAreaRect(contour)[1])
        solidity = area / rect_area if rect_area else 0
        if solidity < MIN_SOLIDITY or area / (w * h) < MIN_FILL:
            return
        sx, sy = int(w * 0.1), int(h * 0.1)
        roi = blur[y + sy : y + h - sy, x + sx : x + w - sx]
        if roi.size < 50:
            return
        lo, hi = np.percentile(roi, [5, 95])
        if hi - lo < min_contrast:
            return
        roi_dark = roi < (lo + hi) / 2
        dark_frac = float(roi_dark.mean())
        if not min_dark <= dark_frac < 0.92:
            return
        dark_sides, usable = _side_score(blur, x, y, w, h)
        if usable >= 2 and dark_sides == 0:
            return
        if strict_sides and usable and dark_sides < (usable + 1) // 2:
            return  # a real card has the dark obstacle face on its left/right
        # agreement of the usable bands, discounted when few bands could be checked
        side_factor = (0.5 + 0.5 * dark_sides / usable) * (0.6 + 0.4 * usable / 3) if usable else 0.5
        centered = 1.0 if _symbol_centered(roi_dark) else 0.5
        squareness = min(w, h) / max(w, h)
        size = min(area / self.frame_area, 0.1) ** 0.25  # prefer the larger of two valid cards, up to 10% of the frame
        glare = 1.0 if hi - lo >= MIN_CONTRAST else 0.5
        score = solidity * squareness * side_factor * centered * size * glare
        if self.debug is not None:
            self.debug.append((round(float(score), 3), (x, y, w, h), round(solidity, 2), round(dark_frac, 2), f'{dark_sides}/{usable}', centered, round(float(hi - lo))))
        if self.best is None or score > self.best[0]:
            self.best = (score, (x, y, w, h))


def detect_card(image, debug=None):
    """Find the symbol card in a BGR image.

    Returns ((cx, cy, w, h), "card") with YOLO-normalized coords, or None when
    no plausible card was found (junk frame, motion blur, card cut off). Pass a
    list as ``debug`` to collect every scored candidate.
    """
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 50, 150), np.ones((3, 3), np.uint8))
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scorer = _Scorer(blur)
    scorer.debug = debug

    # 1. global thresholds: works when the card is the brightest thing around
    for mask in _masks(blur, edges, sorted({int(otsu), *THRESHOLDS})):
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            scorer.consider(c, min_contrast=20, min_dark=0.0)

    # 2. box first: the card is only bright *relative to* the dark obstacle face it
    #    sits on, so threshold locally inside each large dark blob's bounding box
    _, dark = cv2.threshold(blur, DARK_FACE, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    for c in cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) < MIN_FACE_FRAC * W * H:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        m = max(4, int(0.05 * max(bw, bh)))
        x0, y0, x1, y1 = max(0, bx - m), max(0, by - m), min(W, bx + bw + m), min(H, by + bh + m)
        crop = blur[y0:y1, x0:x1]
        if crop.size < 400:
            continue
        hull = np.zeros_like(crop)
        cv2.fillConvexPoly(hull, cv2.convexHull(c) - (x0, y0), 255)
        inside = crop[hull > 0]
        local_otsu, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        levels = sorted({int(local_otsu), int(local_otsu * 0.8), int(local_otsu * 1.2)})
        for mask in _masks(crop, edges[y0:y1, x0:x1], levels):
            mask[hull == 0] = 0
            for cc in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                scorer.consider(cc, x0, y0, min_contrast=20, strict_sides=False, min_dark=0.0)

    if scorer.best is not None:
        return _normalized_box(*scorer.best[1], W, H, PAD_FRAC), "card"

    return None


def _normalized_box(x, y, w, h, frame_w, frame_h, pad):
    x0 = max(0.0, x - w * pad)
    y0 = max(0.0, y - h * pad)
    x1 = min(float(frame_w), x + w * (1 + pad))
    y1 = min(float(frame_h), y + h * (1 + pad))
    return (
        (x0 + x1) / 2 / frame_w,
        (y0 + y1) / 2 / frame_h,
        (x1 - x0) / frame_w,
        (y1 - y0) / frame_h,
    )


def format_label(class_index, box):
    cx, cy, w, h = box
    return f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def write_sidecar_label(image_path, image_id):
    """Detect the card in image_path and write <stem>.txt next to it.

    Used by photobooth.py right after capture. Returns (box, method) or None.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    result = detect_card(image)
    if result is None:
        return None
    box, method = result
    label_path = image_path.with_suffix(".txt")
    label_path.write_text(format_label(image_id - CLASS_ID_BASE, box))
    return box, method


def draw_review(image, box, caption, method):
    frame_h, frame_w = image.shape[:2]
    out = image.copy()
    if box is not None:
        cx, cy, w, h = box
        x0 = int((cx - w / 2) * frame_w)
        y0 = int((cy - h / 2) * frame_h)
        x1 = int((cx + w / 2) * frame_w)
        y1 = int((cy + h / 2) * frame_h)
        color = (0, 200, 0) if method == "card" else (0, 165, 255)  # orange = photobooth sidecar
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, caption, (x0, max(20, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(out, f"FAILED {caption}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def parse_sidecar(image_path):
    """Return the sidecar label's (class_index, box) if one exists, else None."""
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.is_file():
        return None
    parts = sidecar.read_text().split()
    if len(parts) != 5:
        return None
    return int(parts[0]), tuple(float(v) for v in parts[1:])


def convert_split(src_split, out_dir, split_name, names, force, review, failures):
    images_dir = out_dir / "images" / split_name
    labels_dir = out_dir / "labels" / split_name
    review_dir = out_dir / "review" / split_name
    for d in (images_dir, labels_dir) + ((review_dir,) if review else ()):
        d.mkdir(parents=True, exist_ok=True)

    stats = {"labeled": 0, "sidecar": 0, "skipped": 0, "failed": 0}
    for class_dir in sorted(p for p in src_split.iterdir() if p.is_dir()):
        match = CLASS_DIR_RE.match(class_dir.name)
        if not match:
            print(f"  ignoring non-class directory {class_dir.name}")
            continue
        image_id = int(match.group(1))
        class_index = image_id - CLASS_ID_BASE
        names[class_index] = class_dir.name

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            out_image = images_dir / image_path.name
            if label_path.exists() and not force:
                stats["skipped"] += 1
                if not out_image.exists():
                    shutil.copy2(image_path, out_image)
                continue

            method = None
            sidecar = parse_sidecar(image_path)
            if sidecar is not None and not force:
                class_index_used, box = sidecar
                method = "sidecar"
                stats["sidecar"] += 1
            else:
                image = cv2.imread(str(image_path))
                result = detect_card(image) if image is not None else None
                if result is None:
                    stats["failed"] += 1
                    failures.append(image_path)
                    if review:
                        img = image if image is not None else np.zeros((100, 400, 3), np.uint8)
                        cv2.imwrite(str(review_dir / image_path.name),
                                    draw_review(img, None, class_dir.name, None))
                    continue
                box, method = result
                class_index_used = class_index

            label_path.write_text(format_label(class_index_used, box))
            if not out_image.exists() or force:
                shutil.copy2(image_path, out_image)
            stats["labeled"] += 1
            if review:
                image = cv2.imread(str(image_path))
                cv2.imwrite(str(review_dir / image_path.name),
                            draw_review(image, box, f"{class_dir.name} ({method})", method))
    return stats


def write_data_yaml(out_dir, names):
    if not names:
        return
    max_index = max(names)
    val = "images/val"
    if not any((out_dir / "images" / "val").glob("*")):
        print("val split is empty (no dataset/test captures yet) - data.yaml will validate on train")
        val = "images/train"
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        f"val: {val}",
        "",
        "names:",
    ]
    for i in range(max_index + 1):
        # quoted: bare 11_1 would parse as YAML int 111 (underscore separator)
        lines.append(f'  {i}: "{names.get(i, str(i + CLASS_ID_BASE))}"')
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate YOLO detection labels for the symbol-card dataset")
    parser.add_argument("--dataset", type=Path, default=script_dir / "dataset",
                        help="ImageFolder dataset root with train/ and test/ (default: ./dataset)")
    parser.add_argument("--out", type=Path, default=script_dir / "dataset_yolo",
                        help="output YOLO dataset root (default: ./dataset_yolo)")
    parser.add_argument("--force", action="store_true",
                        help="re-run detection even where labels/sidecars already exist")
    parser.add_argument("--no-review", dest="review", action="store_false",
                        help="skip writing annotated review images")
    args = parser.parse_args()

    splits = [("train", "train"), ("test", "val")]
    names, failures = {}, []
    any_split = False
    for src_name, split_name in splits:
        src_split = args.dataset / src_name
        if not src_split.is_dir():
            continue
        any_split = True
        print(f"{src_name}/ -> {split_name}/")
        stats = convert_split(src_split, args.out, split_name, names,
                              args.force, args.review, failures)
        print(f"  labeled {stats['labeled']} (sidecar {stats['sidecar']})"
              f", skipped {stats['skipped']}, failed {stats['failed']}")
    if not any_split:
        print(f"No train/ or test/ under {args.dataset}", file=sys.stderr)
        return 1

    write_data_yaml(args.out, names)
    print(f"Wrote {args.out / 'data.yaml'}")
    if args.review:
        print(f"Review images (boxes drawn) in {args.out / 'review'}")
    if failures:
        print(f"\n{len(failures)} images could not be auto-labeled:")
        for path in failures:
            print(f"  {path}")
        print("Label these by hand or recapture them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Auto-label SC2079 symbol-card images with YOLO bounding boxes.

Detection is classical CV: the symbol card is a bright (white) region with a
high-contrast symbol inside. We find it with Otsu thresholding + contours, and
fall back to boxing the dark symbol blob directly when the card outline can't
be isolated (e.g. card fills the frame or blends into a bright background).

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
PAD_FRAC = 0.03  # padding added around the detected card, per side

# candidate filters, as fractions of the frame
MIN_AREA_FRAC = 0.002
MAX_AREA_FRAC = 0.9
MIN_ASPECT, MAX_ASPECT = 0.25, 4.0
MIN_SOLIDITY = 0.7  # contour area / min-area-rect area: cards are rectangular
MIN_CONTRAST = 60
# a single global Otsu threshold merges the card into other bright regions in
# busy scenes, so sweep fixed levels too and let scoring pick the best blob
THRESHOLDS = (150, 180, 210)


def detect_card(image):
    """Find the symbol card in a BGR image.

    Returns ((cx, cy, w, h), method) with YOLO-normalized coords, where method
    is "card" (white card outline) or "symbol" (fallback: dark symbol blob),
    or None if nothing plausible was found.
    """
    frame_h, frame_w = image.shape[:2]
    frame_area = frame_w * frame_h
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best = None  # (score, (x, y, w, h))
    for threshold in {int(otsu), *THRESHOLDS}:
        _, bright = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not MIN_AREA_FRAC * frame_area < area < MAX_AREA_FRAC * frame_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not MIN_ASPECT < w / h < MAX_ASPECT:
                continue
            rect_area = np.prod(cv2.minAreaRect(contour)[1])
            solidity = area / rect_area if rect_area else 0
            if solidity < MIN_SOLIDITY:
                continue
            roi = blur[y : y + h, x : x + w]
            lo, hi = np.percentile(roi, [5, 95])
            if hi - lo < MIN_CONTRAST:  # no high-contrast symbol inside
                continue
            dark_frac = np.mean(roi < (lo + hi) / 2)
            if not 0.03 < dark_frac < 0.55:
                continue
            score = solidity * np.sqrt(area / frame_area)
            if best is None or score > best[0]:
                best = (score, (x, y, w, h))
    if best is not None:
        return _normalized_box(*best[1], frame_w, frame_h, PAD_FRAC), "card"

    # fallback: box the dark symbol directly and pad generously toward card size
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [
        cv2.boundingRect(c)
        for c in contours
        if 0.002 * frame_area < cv2.contourArea(c) < 0.5 * frame_area
    ]
    if not candidates:
        return None
    x, y, w, h = max(candidates, key=lambda r: r[2] * r[3])
    return _normalized_box(x, y, w, h, frame_w, frame_h, pad=0.15), "symbol"


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
        color = (0, 200, 0) if method == "card" else (0, 165, 255)
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

    stats = {"labeled": 0, "sidecar": 0, "fallback": 0, "skipped": 0, "failed": 0}
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
                if method == "symbol":
                    stats["fallback"] += 1

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
        print(f"  labeled {stats['labeled']}"
              f" (sidecar {stats['sidecar']}, fallback-box {stats['fallback']})"
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

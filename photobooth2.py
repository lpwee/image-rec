#!/usr/bin/env python3
import argparse
import re
import sys
import time
from pathlib import Path
import cv2
import numpy as np

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
except ImportError:
    PiCamera = None

# Category definitions
CATEGORIES = {}
for _id, _d in enumerate("123456789", start=11):
    CATEGORIES[_id] = (_d, f"digit {_d}")
for _id, _c in enumerate("ABCDEFGH", start=20):
    CATEGORIES[_id] = (_c, f"letter {_c}")
for _id, _c in enumerate("STUVWXYZ", start=28):
    CATEGORIES[_id] = (_c, f"letter {_c}")
CATEGORIES[36] = ("up", "up arrow")
CATEGORIES[37] = ("down", "down arrow")
CATEGORIES[38] = ("right", "right arrow")
CATEGORIES[39] = ("left", "left arrow")
CATEGORIES[40] = ("stop", "stop (bullseye)")

def resolve_category(text):
    text = text.strip().lower()
    if not text:
        return None
    if text.isdigit() and int(text) in CATEGORIES:
        cat_id = int(text)
        return cat_id, CATEGORIES[cat_id][0]
    for cat_id, (slug, desc) in CATEGORIES.items():
        if text == slug.lower():
            return cat_id, slug
    matches = [(cat_id, slug) for cat_id, (slug, desc) in CATEGORIES.items() if text in desc.lower()]
    if len(matches) == 1:
        return matches[0]
    return None

def print_menu():
    ids = sorted(CATEGORIES)
    columns = [ids[i : i + 10] for i in range(0, len(ids), 10)]
    print("\nCategories:")
    for row in range(max(len(col) for col in columns)):
        cells = []
        for col in columns:
            if row < len(col):
                cat_id = col[row]
                slug, desc = CATEGORIES[cat_id]
                cells.append(f"{cat_id}  {desc:<18}")
            else:
                cells.append(" " * 22)
        print("   " + "".join(cells))
    print()

def next_index(cat_dir, cat_id):
    pattern = re.compile(rf"^{cat_id}_(\d+)\.jpe?g$", re.IGNORECASE)
    highest = 0
    if cat_dir.is_dir():
        for entry in cat_dir.iterdir():
            match = pattern.match(entry.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1

def main():
    if PiCamera is None:
        print("This script must run on a Raspberry Pi with picamera installed.", file=sys.stderr)
        return 1

    print_menu()
    category_input = input("Choose a category ID or name to start: ")
    resolved = resolve_category(category_input)
    if not resolved:
        print("Invalid category. Exiting.")
        return 1

    cat_id, slug = resolved
    dataset_dir = Path(__file__).resolve().parent / "dataset" / "train"
    cat_dir = dataset_dir / f"{cat_id}_{slug}"
    cat_dir.mkdir(parents=True, exist_ok=True)
    index = next_index(cat_dir, cat_id)

    # Initialize Camera
    camera = PiCamera()
    camera.resolution = (640, 640)
    camera.framerate = 15
    raw_capture = PiRGBArray(camera, size=(640, 640))
    time.sleep(1.0) # Camera Warmup

    print(f"\nCapturing for category: {CATEGORIES[cat_id][1]}")
    print("--------------------------------------------------")
    print("How to use the live window:")
    print("  [SPACEBAR]  - Capture and save image")
    print("  [q]         - Quit and close")
    print("--------------------------------------------------")

    session_count = 0
    cv2.namedWindow("Photobooth Preview", cv2.WINDOW_AUTOSIZE)

    try:
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = frame.array
            
            # Show live preview (this standard window is fully visible over VNC!)
            cv2.imshow("Photobooth Preview", image)
            
            raw_capture.truncate(0)
            
            # Watch for keypresses inside the OpenCV window
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(" "): # SPACEBAR pressed
                path = cat_dir / f"{cat_id}_{index:04d}.jpg"
                while path.exists():
                    index += 1
                    path = cat_dir / f"{cat_id}_{index:04d}.jpg"
                
                # Save the current frame directly to disk
                cv2.imwrite(str(path), image)
                session_count += 1
                print(f"[{session_count}] Saved: {path.name}")
                index += 1
                
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        camera.close()
        print(f"\nFinished! Saved {session_count} images in this session.")

if __name__ == "__main__":
    main()

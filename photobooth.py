#!/usr/bin/env python3
"""Dataset photobooth for SC2079 image classification (picamera).

Pick a category (image IDs 11-40), get a live preview at capture size,
press SPACE to save a still into dataset/<id>_<slug>/ with a
non-conflicting filename like 27_0013.jpg.

Keys while capturing:
    SPACE   capture image
    c       change category (camera keeps running)
    h / ?   show key help
    q / ESC quit

Usage:
    python3 photobooth.py                     # menu, then capture
    python3 photobooth.py --category 27       # jump straight to letter H
    python3 photobooth.py --category stop     # names/descriptions work too
    python3 photobooth.py --list-categories   # print the ID table and exit

Requires the legacy picamera library on the Pi (sudo apt install
python3-picamera, camera enabled via raspi-config). The preview is a GPU
overlay drawn on the Pi's attached display, so it is not visible over SSH —
captures still work without it (--preview none).
If the terminal is ever left in a weird state after a hard crash, run
`stty sane` or `reset`.
"""

import argparse
import re
import select
import sys
import termios
import time
import tty
from pathlib import Path

try:
    from picamera import PiCamera
except ImportError:
    PiCamera = None

# image ID -> (slug, description); slugs name the dataset directories
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
    """Return (id, slug) for an ID, slug, or description substring, else None."""
    text = text.strip().lower()
    if not text:
        return None
    if text.isdigit() and int(text) in CATEGORIES:
        cat_id = int(text)
        return cat_id, CATEGORIES[cat_id][0]
    for cat_id, (slug, desc) in CATEGORIES.items():
        if text == slug.lower():
            return cat_id, slug
    matches = [
        (cat_id, slug)
        for cat_id, (slug, desc) in CATEGORIES.items()
        if text in desc.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(str(cat_id) for cat_id, _ in matches)
        print(f"'{text}' is ambiguous (matches IDs {ids})")
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


def category_dir(dataset_dir, cat_id, slug):
    return dataset_dir / f"{cat_id}_{slug}"


def next_index(cat_dir, cat_id):
    """Smallest unused capture index for this category (1-based)."""
    pattern = re.compile(rf"^{cat_id}_(\d+)\.jpe?g$", re.IGNORECASE)
    highest = 0
    if cat_dir.is_dir():
        for entry in cat_dir.iterdir():
            match = pattern.match(entry.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


class RawTerminal:
    """cbreak mode for single-key reads; Ctrl-C still raises KeyboardInterrupt."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        return False


def read_key(timeout=0.1):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return None


def prompt_category(dataset_dir, initial=None):
    """Resolve initial if given, otherwise prompt with the menu. None = quit."""
    if initial is not None:
        resolved = resolve_category(initial)
        if resolved is None:
            print(f"Unknown category: {initial!r}", file=sys.stderr)
            print_menu()
        else:
            return resolved
    print_menu()
    while True:
        try:
            answer = input("Category (id/name, q to quit): ")
        except EOFError:
            return None
        if answer.strip().lower() in ("q", "quit", "exit"):
            return None
        resolved = resolve_category(answer)
        if resolved is not None:
            return resolved
        print("Not recognised, try again (e.g. 27, H, 'right arrow').")


def start_camera(width, height, preview_mode):
    camera = PiCamera()
    camera.resolution = (width, height)
    if preview_mode == "fullscreen":
        camera.start_preview()
    elif preview_mode == "window":
        # overlay window on the Pi's display at the capture dimensions
        try:
            camera.start_preview(fullscreen=False, window=(40, 40, width, height))
        except Exception as err:
            print(f"Windowed preview unavailable ({err}), continuing without preview")
    if preview_mode == "none":
        print("Preview disabled (captures still work)")
    return camera


HELP_BANNER = "[SPACE] capture  [c] change category  [h] help  [q/ESC] quit"


def capture_loop(camera, dataset_dir, initial_category):
    cat = prompt_category(dataset_dir, initial_category)
    while cat is not None:
        cat_id, slug = cat
        cat_dir = category_dir(dataset_dir, cat_id, slug)
        cat_dir.mkdir(parents=True, exist_ok=True)
        index = next_index(cat_dir, cat_id)
        existing = index - 1
        print(f"Capturing {cat_id} ({CATEGORIES[cat_id][1]}) -> {cat_dir}/ ({existing} existing images)")
        print(HELP_BANNER)

        switch = False
        session = 0
        with RawTerminal():
            while True:
                key = read_key()
                if key is None:
                    continue
                if key == " ":
                    path = cat_dir / f"{cat_id}_{index:04d}.jpg"
                    while path.exists():
                        index += 1
                        path = cat_dir / f"{cat_id}_{index:04d}.jpg"
                    try:
                        camera.capture(str(path))
                    except Exception as err:
                        print(f"Capture failed: {err}")
                        continue
                    if not path.is_file() or path.stat().st_size == 0:
                        print(f"Capture failed: {path} was not written")
                        continue
                    index += 1
                    session += 1
                    print(f"Saved {path} ({session} this session)")
                elif key == "c":
                    switch = True
                    break
                elif key in ("q", "\x1b"):
                    break
                elif key in ("h", "?"):
                    print(HELP_BANNER)
        if not switch:
            break
        cat = prompt_category(dataset_dir)
    print("Bye")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Capture dataset images per category with picamera2")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=script_dir / "dataset" / "train",
        help="split directory to capture into (default: dataset/train; use --dataset dataset/test for test images)",
    )
    parser.add_argument("--category", help="start in this category (id/name), skipping the menu")
    parser.add_argument("--width", type=int, default=640, help="capture width (default 640)")
    parser.add_argument("--height", type=int, default=640, help="capture height (default 640)")
    parser.add_argument(
        "--preview",
        choices=["window", "fullscreen", "none"],
        default="window",
        help="preview overlay on the Pi's display (default: window at capture size)",
    )
    parser.add_argument("--warmup", type=float, default=2.0, help="seconds to let exposure/white balance settle")
    parser.add_argument("--list-categories", action="store_true", help="print the category table and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_categories:
        print_menu()
        return 0
    if PiCamera is None:
        print(
            "picamera is not available. This tool must run on a Raspberry Pi "
            "with python3-picamera installed (sudo apt install python3-picamera) "
            "and the camera enabled (sudo raspi-config).",
            file=sys.stderr,
        )
        return 1
    if not sys.stdin.isatty():
        print("photobooth.py needs an interactive terminal for key input.", file=sys.stderr)
        return 1

    width = args.width // 2 * 2
    height = args.height // 2 * 2
    try:
        camera = start_camera(width, height, args.preview)
    except Exception as err:
        print(
            f"Failed to start camera: {err}\n"
            "Check that no other app is using the camera, the camera is enabled "
            "in raspi-config, and the ribbon cable is seated.",
            file=sys.stderr,
        )
        return 1

    try:
        time.sleep(args.warmup)
        capture_loop(camera, args.dataset, args.category)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

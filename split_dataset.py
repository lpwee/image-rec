#!/usr/bin/env python3
"""Re-split the ImageFolder captures in dataset/ into train/ and test/.

Pools every image of each class from dataset/train and dataset/test, then
moves a fixed, seeded fraction per class into dataset/test (stratified split).
Photobooth sidecar labels (<stem>.txt) travel with their image.

    python3 split_dataset.py                 # 80/20, seed 42
    python3 split_dataset.py --test-frac 0.2 --seed 42

Classes with fewer than --min-class images keep everything in train (one
held-out image of a two-image class tells you nothing).
Run autolabel.py --force afterwards to rebuild dataset_yolo/.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=root / "dataset")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-class", type=int, default=3,
                        help="classes smaller than this go entirely to train")
    args = parser.parse_args()

    train_dir, test_dir = args.dataset / "train", args.dataset / "test"
    if not train_dir.is_dir():
        print(f"missing {train_dir}", file=sys.stderr)
        return 1

    # pool images per class
    pool = {}
    for split_dir in (train_dir, test_dir):
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                for p in class_dir.iterdir():
                    if p.suffix.lower() in IMAGE_EXTS:
                        pool.setdefault(class_dir.name, []).append(p)

    rng = random.Random(args.seed)
    total_train = total_test = 0
    print(f"{'class':<10}{'total':>7}{'train':>7}{'test':>6}")
    for class_name in sorted(pool):
        images = sorted(pool[class_name], key=lambda p: p.name)
        n = len(images)
        n_test = max(1, round(n * args.test_frac)) if n >= args.min_class else 0
        rng.shuffle(images)
        test_set = set(images[:n_test])
        for p in images:
            dest_dir = (test_dir if p in test_set else train_dir) / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in (p, p.with_suffix(".txt")):  # image + optional sidecar
                if f.exists() and f.parent != dest_dir:
                    shutil.move(str(f), str(dest_dir / f.name))
        print(f"{class_name:<10}{n:>7}{n - n_test:>7}{n_test:>6}")
        total_train += n - n_test
        total_test += n_test

    # keep every class dir in test/, even empty ones: Ultralytics numbers classes
    # from the folders present, so a missing folder shifts the test-set indices
    for class_name in pool:
        class_dir = test_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / ".gitkeep").touch()
    print(f"{'total':<10}{total_train + total_test:>7}{total_train:>7}{total_test:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SC2079 symbol-card recognition

Two YOLO26-nano models fine-tuned on our own captures of the 30 symbol cards (image IDs 11-40).

| Model | Notebook | Input | Output | Test result (70 held-out images) |
|---|---|---|---|---|
| Detector (`yolo26n`) | `train.ipynb` | `dataset_yolo/` (needs autolabel) | class + bounding box | mAP50 0.964, mAP50-95 0.849, P 0.95 / R 0.96 |
| Classifier (`yolo26n-cls`) | `train_cls.ipynb` | `dataset/` directly | class only | top-1 accuracy 1.000 (70/70) |

Deployment weights: `weights/best.pt` (detector), `weights/best_cls.pt` (classifier).
Prediction class -> image ID: `int(model.names[cls].split("_")[0])`.

## What has been done

### 1. Dataset re-split (`split_dataset.py`)
Photobooth writes straight into `dataset/train` or `dataset/test`, so the original split was whatever
was captured last. `split_dataset.py` pools every image per class from both folders and moves a seeded,
stratified 20% into `dataset/test` (class folders are kept in both splits even when empty, because
Ultralytics numbers classes from the folders present). Current split: 282 train / 70 test, 10 classes.

### 2. Junk captures moved to `dataset/junk/`
Eight captures contain no card at all: both `20_A` images (a group photo), both `21_B` images (a hand
under a desk), `11_0002`, `12_0001`, `13_0001` (no card in frame) and `14_0008` (corrupted frame).
They were moved, not deleted, and nothing reads `dataset/junk/`. Classes A and B therefore have no
data yet and need recapturing. `13_0002` (hand shot) is still in the test split and should go too.

### 3. Auto-labeler rewritten (`autolabel.py`)
The detector needs boxes; the captures only carry a class (folder name). The original labeler took the
largest bright blob with some contrast inside, which in practice was the wall or tabletop: about a third
of all boxes covered more than half the frame, a few boxed the hole inside a digit, and the black stop
card was never found. A detector trained on those labels reached only mAP50 0.69 and predicted
whole-frame boxes.

The new `detect_card()`:
- gathers candidate bright blobs from **global thresholds** (card brighter than everything) and
  **"box first"** (find each large dark blob, i.e. the obstacle face, and threshold locally inside its
  convex hull, which handles dim scenes where the card is only bright relative to the obstacle);
- filters on shape (area, aspect, solidity, fill, not spanning the full frame width);
- scores each survivor on: dark left/right/bottom surround (the obstacle face), a dark symbol centred
  inside, squareness, and a size bonus capped at 10% of the frame so a huge wall region can never
  outscore a real card;
- has **no fallback box**: an image with no plausible card is listed as failed and left out.

Across all 360 images the whole-frame boxes went from 105 to 2 (one of which is a genuine close-up),
tiny boxes from 15 to 1, and the stop card is found on 22/22 instead of 14/22. Only one image
(motion blur) is skipped. Skim `dataset_yolo/review/` after each run; green boxes are auto-detected,
orange come from a photobooth sidecar.

### 4. Classifier notebook added (`train_cls.ipynb`)
Since the folder names already are labels, a classifier can train straight from `dataset/` with no
labeling step. It reaches 100% on the test split by epoch 10. Trade-off: no card location, and a
whole-frame decision is more fragile than a detector when the card is small in a cluttered scene.

### 5. Both models retrained on the cleaned, relabeled data
Detector: 80 epochs, imgsz 640, heavy geometric/colour augmentation, no horizontal flip (it would turn
digits and the left/right arrow cards into the wrong class). Best epoch 65. Runs in `runs/train-6`,
validation in `runs/val-4`. Classifier: 80 epochs, run in `runs/cls-2`.

Per-class detector mAP50 is 0.995 for eight of ten classes (stop card included); `16_6` and `18_8`
are lowest at 0.855.

### Caveats
- The test split is small (70 images) and shot in the same rooms and lighting as train, so both scores
  are optimistic for a new environment. Collect angled / far / other-room captures before trusting them.
- Classes A and B have no data; IDs 22-39 have none at all yet.
- `server.py` does not load a model yet.

## Reproduce
```sh
pip install ultralytics
# Blackwell-class GPUs need a CUDA 12.8 torch build; the stock cu124 wheel has no kernels for them:
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python3 split_dataset.py            # optional: re-split 80/20 per class, seed 42
python3 autolabel.py --force        # dataset/ -> dataset_yolo/ (images, labels, data.yaml, review/)
jupyter nbconvert --to notebook --execute --inplace train.ipynb       # detector, ~6 min on an RTX PRO 4000
jupyter nbconvert --to notebook --execute --inplace train_cls.ipynb   # classifier, ~4 min
```
Training on a fresh machine: delete `dataset_yolo/images`, `labels` and `review` before running
autolabel, otherwise stale files from an older split leak into the new one.

## Files
| Path | Purpose |
|---|---|
| `photobooth.py`, `photobooth2.py`, `camera.py` | capture images into `dataset/<split>/<id>_<slug>/` |
| `split_dataset.py` | stratified train/test re-split |
| `autolabel.py` | ImageFolder -> YOLO detection dataset with auto-generated boxes |
| `train.ipynb` | detector training, validation, export to `weights/best.pt` |
| `train_cls.ipynb` | classifier training, validation, export to `weights/best_cls.pt` |
| `dataset/` | raw captures (`train/`, `test/`, `junk/`) |
| `dataset_yolo/` | generated detection dataset |
| `runs/` | Ultralytics training/validation outputs |
| `server.py` | Flask endpoint stub for the robot |

---

# Original planning notes

## What I have in mind:
1. Take a base model (what kind of model?)
    - Model Pixel Dimensions? 640
2. Retrain classifier head specifically for classifying our images
    - Need multiple angles of the same image, and different distances
    - Apply imge transformations to generate more datasets (flipping, scaling, colour shifts)
3. ???
4. Profit

### Base Model Selection
Looking at state of the art models to build upon,
I am looking out for a few things:
- Model Size (for faster inference, and to fit on our measly 4GB)
- Model Type? (https://docs.ultralytics.com/tasks/obb) draws bounding boxes around objects
- Model Image Dimensions
- Model Architecture

### Dataset Collection / Generattion / Preparation
I have this idea in my head, that we can train a model to recognise and detect the images without looking head-on (e.g drive-by angle is enough to detect) for that we may need to:
- Collect angled datasets, ALOT OF IT.
- Layer Freezing for retraining of classifier head to not catastropihcally collapse features learned from earlier layers

### Misc Non-functional requirements
- Multithreaded performance?
- Image polling rate?

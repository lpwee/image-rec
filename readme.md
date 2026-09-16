# Image-rec task

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


## Autolabel
```py
python3 autolabel.py
# --force (to redo everything)
```
Converts ImageFolder `dataset/` (ResNet-18's standard) to `dataset_yolo` (Ultralytics' Standard).
This is done by keeping the identified labels (from photobooth) and draw bounding boxes (using opencv).


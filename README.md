# PPE Compliance Checker — Pose-Guided Helmet & Vest Detection

A computer-vision safety tool built with [Ultralytics YOLO](https://docs.ultralytics.com/) that
checks whether people in images and video are wearing a **helmet** and a **hi-vis vest**. It
fuses two YOLO tasks — **pose estimation** (to locate each person's head and chest) and
**object detection** (to find the actual PPE items) — so that PPE items are attributed to the
*right person*, not just detected loose in the frame.

Built as the capstone project for **Computer Vision for Developers with Ultralytics**, a 5-day,
30-hour on-site training program delivered by SDAIA Academy via Learning Space.
See [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy).

## What it does

1. A pose model (`yolo26n-pose.pt`) detects every person in a frame and their 17 body keypoints.
2. From those keypoints we derive a **head region** (nose/eyes/ears, expanded upward for where a
   helmet sits) and a **chest region** (shoulders to hips).
3. A PPE detector, fine-tuned on the [Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/),
   finds `helmet`, `no_helmet`, `vest`, and other PPE-related classes in the same frame.
4. Each detected item is matched to the person whose head/chest region it overlaps.
5. Each person is labeled **SAFE** (helmet + vest both found) or **VIOLATION** (either missing).
6. On video, people are tracked (`model.track`, ByteTrack) so the same person keeps a stable ID
   across frames instead of being re-counted every frame.

## Project scope / tasks used

- **Pose estimation** — `yolo26n-pose.pt`, Ultralytics pretrained COCO-pose weights.
- **Object detection** — a YOLO26n detector fine-tuned on Construction-PPE (11 classes).
- **Tracking** — `model.track()` with ByteTrack for the video pipeline.
- **Export** — the fine-tuned detector exported to ONNX for portable inference.

## Datasets

| | Source | Use |
|---|---|---|
| Images | [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) (1,416 images, 11 classes, AGPL-3.0) | Fine-tuning + evaluation of the PPE detector |
| Video | [Sample Videos for Helmet Detection on YOLOv8 (Kaggle)](https://www.kaggle.com/datasets/ayushraj2349/sample-videos-for-helmet-detection-on-yolov8) | End-to-end video pipeline demo |

The image dataset auto-downloads the first time the notebook references `construction-ppe.yaml`
(it ships inside the `ultralytics` package). The video dataset needs a free Kaggle account —
download it manually from the link above and upload it in the notebook (instructions in §1.2),
or use the Kaggle API if you have a token.

## Repository structure

```
.
├── PPE_Compliance_Capstone.ipynb   # the whole project: setup -> train -> infer -> eval -> video -> export
├── README.md
├── requirements.txt
└── .gitignore
```

Trained weights, the dataset, and Ultralytics run folders (`runs_ppe/`, `datasets/`) are
intentionally **not** committed — see `.gitignore`. If you want to share your trained weights,
attach them as a GitHub Release asset or link a Drive folder here instead of committing the
binary.

## How to run

1. Open `PPE_Compliance_Capstone.ipynb` in Google Colab.
2. `Runtime → Change runtime type → GPU` (a free T4 is enough).
3. Run the cells top to bottom:
   - **§0 Setup** installs `ultralytics`, `opencv-python-headless`, `lap`.
   - **§1 Get the data** auto-downloads Construction-PPE, and prompts you to upload the Kaggle
     video zip (or use the Kaggle API cell if you have a token).
   - **§2 Training** fine-tunes the detector (`model.train`) — adjust `epochs`/`imgsz`/`batch`
     for your GPU and time budget.
   - **§3 Inference demo**, **§4 Evaluation** (`model.val`, mAP/confusion matrix),
     **§5 Video pipeline** (`model.track` + OpenCV, produces `ppe_compliance_output.mp4`),
     **§6 Export** (`model.export(format="onnx")`) all run in order using the model you trained
     in §2.
4. Keep the executed output when you save/commit the notebook — that's the evidence of a real run.

## Expected output

- Two side-by-side plots per demo image: pose keypoints and PPE detection boxes (§3).
- Printed mAP50 / mAP50-95 / precision / recall plus a confusion-matrix image (§4).
- An annotated output video (`ppe_compliance_output.mp4`) with a green **SAFE** or red
  **VIOLATION** box and a stable track ID drawn on every person, plus a printed summary
  (unique people tracked, how many were ever flagged) (§5).
- An `.onnx` file next to the trained weights (§6).

## Results (from the executed run)

Trained YOLO26n on Construction-PPE, 57 epochs (early-stopped, best checkpoint at epoch 42),
~23 minutes on a Colab T4:

| Metric | Value |
|---|---|
| mAP50 | 0.520 |
| mAP50-95 | 0.258 |
| Precision (mean) | 0.657 |
| Recall (mean) | 0.499 |

The "positive" PPE classes (`helmet`, `vest`, `boots`, `gloves`, `goggles`, `Person`) all scored
AP50 0.66-0.87. The "negative" violation classes (`no_helmet`, `no_goggle`, `no_gloves`,
`no_boots`) scored much lower (AP50 0.01-0.37) due to far fewer training examples — see the
notebook's §4 for the full breakdown and confusion-matrix analysis. On the sample video, the
pipeline tracked 3 people and correctly flagged all 3 as missing their helmet (worn in-hand, not
on the head) while correctly reading their vests as compliant — see §5.

## Known limitations

- The Construction-PPE dataset has a `no_helmet` class but no `no_vest` class, so "missing vest"
  is *inferred* (no vest detected near the chest region) rather than directly classified — see
  the notebook's evaluation section for how this affects false negatives.
- Head/chest regions are geometric approximations from keypoints, not exact anatomical boxes;
  they can misfire when a person is heavily occluded, facing directly away from the camera (no
  face keypoints), or very close to another person.

## License

Code in this repository is released under the MIT License. The Construction-PPE dataset and Ultralytics
YOLO weights are licensed **AGPL-3.0** — see [Ultralytics licensing](https://www.ultralytics.com/license)
before any commercial use, and cite the dataset as:

> Dalvi, M., Singh, N., Bhingarde, S., Chalke, K. (2025). *Construction-PPE: Personal Protective
> Equipment Detection Dataset* (v1.0.0). Ultralytics.

# Downloading the Roboflow Thermal Dogs and People Dataset

Recommended dataset:
- https://public.roboflow.com/object-detection/thermal-dogs-and-people
- https://universe.roboflow.com/joseph-nelson/thermal-dogs-and-people

## Option A (Fastest): Download via Browser (Roboflow Public)

1. Open the dataset page.
2. Click **Downloads** → choose a format (e.g., **COCO JSON** or **Pascal VOC**).
3. Download and unzip the dataset locally.

Put images into:
```
data/full_dataset/images/
```

(Optional) If you downloaded COCO format, keep annotations at:
```
data/full_dataset/annotations/instances_train.json
data/full_dataset/annotations/instances_valid.json
data/full_dataset/annotations/instances_test.json
```

## Option B: Roboflow API (Requires a Free API Key)

If you have a Roboflow account and API key, you can programmatically download the dataset using
the official `roboflow` Python SDK. This is optional and not required for this assignment.

---
## After Download

Run the classical pipeline on a folder:
```bash
python thermal_animal_segmentation.py --input data/full_dataset/images --output_dir outputs/classical_full
```

Then run SAM2 comparison:
```bash
python sam2_comparison.py --input data/full_dataset/images --output_dir outputs/sam2_full
```

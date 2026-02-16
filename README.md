# Thermal Animal Boundary Extraction (OpenCV-only) + SAM2 Comparison

**Course:** CSc 8830 — Computer Vision (Graduate)  
**Module:** 4  
**Task:** Find the *exact boundaries* (precise contours / mask) of an **animal** in **thermal infrared** images using **classical OpenCV** (no ML / no deep learning), and **compare** results with **SAM2**.

> ✅ The **core** solution is `thermal_animal_segmentation.py` and uses **only OpenCV + NumPy**.  
> 🧪 `sam2_comparison.py` is *separate* and may use SAM2 (deep learning) **only for comparison**, as required by the assignment.

---
## Repository Structure

```
thermal-animal-boundary-opencv/
├─ thermal_animal_segmentation.py      # OpenCV-only pipeline (main deliverable)
├─ sam2_comparison.py                  # SAM2 comparison (optional, heavy deps)
├─ requirements.txt                    # classical deps
├─ requirements-sam2.txt               # SAM2 deps
├─ README.md
├─ report/
│  ├─ report.md                        # Markdown report (convert to PDF)
│  └─ figures/                         # place screenshots here
├─ data/
│  ├─ sample_inputs/                   # small demo images (lightweight)
│  ├─ sample_outputs/                  # (optional) curated outputs for README/report
│  └─ full_dataset/                    # place Roboflow dataset here (not committed)
├─ outputs/
│  ├─ classical/                       # generated outputs (OpenCV)
│  └─ sam2/                            # generated outputs (SAM2)
└─ tools/
   └─ download_dataset_instructions.md
```

---
## Dataset

Recommended dataset: **Roboflow Thermal Dogs and People** (203 images)  
- Public page: https://public.roboflow.com/object-detection/thermal-dogs-and-people  
- Universe project: https://universe.roboflow.com/joseph-nelson/thermal-dogs-and-people

This repo includes a **small** `data/sample_inputs/` set for quick testing. For the assignment submission,
you should download the full dataset and run the scripts on **5–8 varied images** (close/far, single/multiple, different poses).

See: `tools/download_dataset_instructions.md`

---
## Installation (Classical OpenCV-only)

```bash
git clone <YOUR_GITHUB_REPO_URL>.git
cd thermal-animal-boundary-opencv

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell

pip install -r requirements.txt
```

---
## Run (Classical OpenCV-only)

**Single image**
```bash
python thermal_animal_segmentation.py \
  --input data/sample_inputs/01_dog_crop.png \
  --output_dir outputs/classical
```

**Folder batch**
```bash
python thermal_animal_segmentation.py \
  --input data/sample_inputs \
  --output_dir outputs/classical
```

Outputs are saved under:
```
outputs/classical/<image_stem>/
  *_01_original.png
  *_03_threshold.png
  *_04_morph.png
  *_05_mask.png
  *_06_overlay.png
  *_07_side_by_side.png
outputs/classical/run_summary.json
```

---
## Method Summary (OpenCV-only)

Pipeline (as required):
1. **Grayscale** conversion
2. **Gaussian blur** (noise suppression; stabilizes histogram)
3. **Otsu threshold** (or optional adaptive threshold) to separate warm body from cool background
4. **Morphology**: opening + closing (+ mild dilation) to remove speckle noise and fill holes
5. **Contour extraction**
6. **Area + shape filtering**
7. **Final mask + contour overlay** + side-by-side diagnostics

Thermal images often provide strong foreground/background separation due to temperature differences,
making classical thresholding surprisingly effective in many cases.

---
## SAM2 Comparison (Optional)

SAM2 is **not allowed** in the main solution, but is used here for comparison.

### Install SAM2 dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-sam2.txt
```

> If you hit CUDA / torch issues, install a PyTorch build matching your system from https://pytorch.org.

### Run SAM2 comparison
```bash
python sam2_comparison.py \
  --input data/sample_inputs \
  --output_dir outputs/sam2 \
  --sam2_model facebook/sam2-hiera-large
```

Outputs are saved under:
```
outputs/sam2/<image_stem>/
  *_02_classical_overlay.png
  *_03_sam2_overlay.png
  *_04_sam2_mask.png
  *_05_side_by_side.png
outputs/sam2/run_summary.json
```

---
## Producing the Report PDF

The report is written in Markdown at `report/report.md`.  
Convert to PDF using any of:
- VS Code “Markdown PDF”
- Typora export
- Pandoc (`pandoc report.md -o report.pdf`)

Replace placeholders like:
- `<YOUR_GITHUB_REPO_URL>`
- `[Insert Screenshot ...]`

---
## Demo Video

See the “Video Demonstration Instructions” section at the end of `report/report.md` for a 3–5 minute script.

---
## License

Code: MIT (suggested). Dataset license: see Roboflow/Universe listing for details.

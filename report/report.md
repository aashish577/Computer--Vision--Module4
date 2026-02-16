# CSc 8830 — Computer Vision  
## Module 4 Assignment Report  
### Thermal Animal Boundary Extraction with Classical OpenCV + SAM2 Comparison

**Student:** <Your Name>  
**Instructor:** <Instructor Name>  
**Date:** <Submission Date>  
**GitHub Repository:** <YOUR_GITHUB_REPO_URL>

---

## Abstract

This project implements a classical (non-ML) image processing pipeline using OpenCV to extract the precise
boundary (pixel-wise mask and contour) of an animal in thermal infrared imagery. Thermal imagery often exhibits
strong contrast between warm bodies and cooler backgrounds, making threshold-based segmentation viable.
Results are compared qualitatively and quantitatively (where feasible) against Meta’s Segment Anything Model 2 (SAM2),
which provides stronger boundary fidelity but requires deep learning and significantly higher computational resources.

---

## 1. Introduction and Problem Statement

Thermal infrared cameras map emitted radiation to pixel intensities (often displayed using a false-color palette).
In many scenes, animals appear as relatively **warm** regions compared to the surrounding environment.

**Goal:** Implement an OpenCV-only script that finds the **exact boundaries** of an animal (precise contours/mask)
in thermal images. Deep learning and ML models are not allowed for the primary solution. A separate comparison
with SAM2 is required.

**Core constraints:**
- Use only OpenCV (classical CV) for the main solution.
- Provide a working demo and record a short screen-capture video of the system.

---

## 2. Dataset

Recommended dataset: **Roboflow Thermal Dogs and People** (203 images)  
Source: https://public.roboflow.com/object-detection/thermal-dogs-and-people  
Universe page: https://universe.roboflow.com/joseph-nelson/thermal-dogs-and-people

The dataset contains thermal images captured at varying distances, orientations, and backgrounds. Although the
dataset provides **bounding boxes** (object detection labels), it does **not** provide pixel-accurate segmentation
ground truth. Therefore, quantitative evaluation uses weak proxies (e.g., box overlap, area consistency) and
comparisons between classical masks and SAM2 masks.

**Images used in this report:** (select 5–8 images with variety)
1. <Image 1 name> — close dog, clear contrast
2. <Image 2 name> — far dog, small target
3. <Image 3 name> — multiple animals or dog+person
4. <Image 4 name> — partial occlusion / challenging background
5. <Image 5 name> — different pose / orientation
6. (Optional) <Image 6–8>

---

## 3. Methodology (Classical OpenCV Pipeline)

The implemented pipeline is intentionally classical, interpretable, and aligned with the assignment requirements:

1. **Grayscale conversion**  
   Thermal images in the Spectra palette are pseudo-color; converting to grayscale collapses color to intensity.

2. **Gaussian blur**  
   A Gaussian kernel reduces sensor noise and suppresses small isolated hot pixels, stabilizing threshold selection.

3. **Otsu thresholding (or Adaptive thresholding)**  
   Many thermal scenes are approximately bimodal (warm foreground vs cool background). Otsu’s method chooses a
   threshold that minimizes intra-class variance:

   Let the histogram be split at threshold \(t\) into classes \(C_0\) and \(C_1\). Otsu selects \(t\) that minimizes:
   \[
     \sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)
   \]
   equivalently maximizing between-class variance \(\sigma_b^2(t)\).

   In practice, the algorithm uses `cv2.threshold(..., THRESH_OTSU)`.

4. **Morphological operations**  
   - **Opening** removes small speckles (erosion → dilation)  
   - **Closing** fills holes inside the animal blob (dilation → erosion)  
   - **Dilation (mild)** connects fragmented warm regions (legs/ears) into a coherent component

   Elliptical structuring elements work well for organic shapes.

5. **Contour extraction**  
   External contours (`RETR_EXTERNAL`) provide candidate object boundaries.

6. **Contour filtering**  
   Candidate contours are filtered using:
   - **Area thresholds** (min/max)
   - **Solidity** (area / convex hull area) to reject extremely fragmented blobs
   - **Circularity** to reject thin line-like noise regions

7. **Mask and overlay generation**
   The selected contour(s) are filled to form a binary mask and overlaid on the original image with alpha blending.

---

## 4. Implementation Details

**Key scripts**
- `thermal_animal_segmentation.py` — OpenCV-only implementation (main deliverable)
- `sam2_comparison.py` — SAM2 comparison script (separate dependencies)

**Inputs**
- Single image or a folder of images via `argparse`

**Outputs (per input image)**
- Original image
- Threshold result
- Morphology result
- Binary mask
- Final overlay with contour
- Side-by-side diagnostic strip

**Robustness notes**
- Auto-inverts threshold if the “foreground” occupies too much area (common failure mode when polarity flips)
- Handles arbitrary image sizes and orientations
- Stores run summary JSON including processing time and mask area

---

## 5. Results (OpenCV-only)

Provide 5–8 varied examples. For each, include the side-by-side strip produced by the script.

**Example 1: <Image 1 name>**  
[Insert Screenshot 1: side-by-side (Original, Gray, Threshold, Morphology, Overlay)]

Observations:
- Boundary adherence: <excellent / moderate / weak>
- Typical failure modes: <e.g., merges dog+person, misses limbs, holes in torso>

**Example 2: <Image 2 name>**  
[Insert Screenshot 2: side-by-side]

**Example 3: <Image 3 name>**  
[Insert Screenshot 3: side-by-side]

**Example 4: <Image 4 name>**  
[Insert Screenshot 4: side-by-side]

**Example 5: <Image 5 name>**  
[Insert Screenshot 5: side-by-side]

(Add more as needed.)

---

## 6. Comparison with SAM2

SAM2 is a foundation model for promptable segmentation. Here it is used only for comparison, not for the core solution.

**Prompt strategy**
- Use the OpenCV mask to derive a **box prompt** around the candidate animal region.
- Optionally sample a few **positive points** inside the OpenCV mask and **negative points** outside to refine.

**Side-by-side comparisons**
- For each test image, include:
  [Insert SAM2 Comparison Screenshot: Original | Classical Overlay | SAM2 Overlay]

**Qualitative summary**
- **SAM2 strengths:** sharper boundaries, better separation when warm objects are close together, more robust to clutter.
- **SAM2 weaknesses:** heavy dependencies (PyTorch + checkpoint), slower on CPU, reveals less interpretability than classical pipeline.
- **OpenCV strengths:** fast, lightweight, explainable, no training, works well when thermal contrast is strong.
- **OpenCV weaknesses:** can merge multiple warm objects, can fail when background has warm regions or contrast is weak.

---

## 7. Quantitative Metrics (Where Possible)

Because the dataset provides bounding boxes rather than pixel masks, we use pragmatic proxy metrics:

1. **Runtime per image (ms)**
   - Classical pipeline: read from `outputs/classical/run_summary.json`
   - SAM2 pipeline: read from `outputs/sam2/run_summary.json`

2. **Mask area (pixels)**
   - Measures stability across images (large swings can indicate threshold instability).

3. **Classical vs SAM2 mask IoU**
   - Since SAM2 is generally a stronger segmenter, the IoU provides a rough measure of how close the classical method is:
     \[
       IoU = \frac{|M_{cv} \cap M_{sam2}|}{|M_{cv} \cup M_{sam2}|}
     \]

4. **Bounding box proxy overlap (optional)**
   If COCO boxes are available, compute:
   - IoU between predicted mask’s bounding rect and GT bounding rect
   - Fraction of predicted mask inside the GT dog box

**Table template**
| Image | Classical ms | SAM2 ms | Mask Area (px) | IoU (CV vs SAM2) | Notes |
|------|--------------:|--------:|---------------:|-----------------:|------|
| img1 | <...> | <...> | <...> | <...> | <...> |
| img2 | <...> | <...> | <...> | <...> | <...> |

---

## 8. Limitations and Future Improvements

**Limitations**
- Warm background objects (cars, lights, heated surfaces) can produce false positives.
- Multiple warm objects (person + dog) may be merged into one blob after morphology.
- Spectra pseudo-color mapping is not guaranteed linear in temperature; grayscale conversion is a heuristic.

**Possible improvements (still classical)**
- Use HSV or Lab channels to better isolate “hot” colors in Spectra palette (e.g., threshold on V or a-b).
- Combine thresholding with edge evidence (Canny edges + active contour / morphological snakes).
- Use connected components with region statistics to reject human-like tall blobs when goal is dog-only.
- Multi-scale processing for small, far animals.

---

## 9. Conclusion

A pure OpenCV pipeline (grayscale → Gaussian blur → Otsu/adaptive threshold → morphology → contours)
can produce accurate animal boundaries in many thermal images due to strong thermal contrast and relatively simple
intensity distributions. SAM2 typically produces tighter boundaries and better separation in complex scenes, but
is computationally heavier and violates the “no deep learning” constraint for the primary solution.

---

## References

1. Roboflow Thermal Infrared dataset post: https://blog.roboflow.com/thermal-infrared-dataset-computer-vision/  
2. Roboflow Thermal Dogs and People dataset page: https://public.roboflow.com/object-detection/thermal-dogs-and-people  
3. Meta AI SAM2 project: https://ai.meta.com/research/sam2/

---

## Video Demonstration Instructions (3–5 minutes)

**Goal:** Show your code is organized, reproducible, and works; explain the classical pipeline clearly; demonstrate SAM2 comparison.

### Recommended recording setup
- Tool: OBS Studio (1080p, 30 FPS)
- Capture: screen + terminal + (optional) file browser
- Keep terminal font large and readable.

### Suggested timeline / narration

**0:00–0:20 — Intro**
- “This is my Module 4 assignment for CSc 8830: thermal animal boundary extraction using OpenCV only.”
- Mention dataset and constraint: no ML/DL in main script.

**0:20–0:50 — Repo overview**
- Show repository structure.
- Point out main scripts:
  - `thermal_animal_segmentation.py` (OpenCV-only)
  - `sam2_comparison.py` (comparison only)
- Mention where outputs are saved.

**0:50–2:10 — Run classical pipeline**
- Run on a single image first:
  ```bash
  python thermal_animal_segmentation.py --input <image> --output_dir outputs/classical
  ```
- Open the generated outputs folder and show:
  - threshold image
  - morphology result
  - final overlay + contour
  - side-by-side diagnostic strip
- Narration points:
  - Why grayscale + blur
  - Why Otsu works well on thermal (bimodal distribution)
  - Why morphology is necessary (noise, holes, fragmentation)
  - How contours yield precise boundaries

**2:10–3:10 — Run batch on 5–8 images**
- Run on a folder:
  ```bash
  python thermal_animal_segmentation.py --input <folder> --output_dir outputs/classical
  ```
- Quickly flip through 3–4 side-by-side strips.
- Mention at least one “easy” case and one “hard” case (e.g., dog near person, far target).

**3:10–4:30 — SAM2 comparison**
- Run:
  ```bash
  python sam2_comparison.py --input <same folder> --output_dir outputs/sam2
  ```
- Show side-by-side: Original | Classical | SAM2
- Discuss:
  - Boundary quality improvements
  - Cases where SAM2 separates person vs dog better
  - Tradeoffs: compute cost, dependency weight, not allowed for main solution

**4:30–5:00 — Wrap up**
- Summarize results + limitations + potential classical improvements.
- Remind where the PDF report and outputs are.


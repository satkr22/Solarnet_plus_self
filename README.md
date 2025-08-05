# SolarNet‑Plus (Rooftop & Superstructure Segmentation)

**A reproduction and adaptation of SolarNet+ for rooftop footprint & superstructure extraction only**  
*(solar potential calculation module **not included**) *

---

## Overview

This repository contains my **adapted implementation** of the **SolarNet+** framework (from the official paper and GitHub) focused on **roof footprint and superstructure semantic segmentation** using the RID dataset.

While the original SolarNet+ includes solar potential estimation, I reproduced the parts dealing with:
- **Roof segmentation** (classifying building roof vs. background)
- **Roof superstructures segmentation** (e.g., chimneys, skylights, HVAC units)

I **did not implement** the solar potential calculation pipeline.


## What I Did

- **Dataset preparation**  
  - Used the official RID dataset (roof + superstructure annotations) from the original SolarNet+ repo.  
  - **Refactored** and **aligned** image-mask pairs to match model input (e.g. resizing, organizing channels).

- **Code adaptation & training**  
  - Modified `transolarnet.py` and training pipeline to run on my computing environment.  
  - Ensured compatibility (e.g. label channels, class counts, input-output paths).

- **Inference pipeline**  
  - Validated predictions on both **patch-based** and **full-image** workflows (`predict_patch.py` & `predict_tif.py`).

- **Logging & visualization**  
  - Corrected minor bugs (file paths, label mismatches).  
  - Logged IoU metrics on segmentation.  
  - Visualized segmented roof outlines and superstructures.

---

## Not Implemented

- **Solar potential estimation (e.g., PV generation, azimuth/pitch inference, area-based output)**  
  - Although the original SolarNet+ repo includes `pvcode/cal_pv.py` to compute solar potential using segmentation outputs, I **did not run or customize** it.  
  - This repo focuses purely on the segmentation tasks.

---


Output maps include:

-Roof segmentation mask (2 classes)

-Roof superstructure mask (multiple classes or binary)

## Results & Highlights
Successfully reproduced roof footprint and superstructure segmentation using SolarNet+ architecture.

Achieved segmentation **IoU of 0.76** (comparable to original performance reported).

Learned how dataset alignment, mask channel structure, and class imbalance affect final results.

This experience served as a stepping stone toward implementing a more robust HRNet + OCR pipeline later.

## References
Qingyu Li, Sebastian Krapf, Lichao Mou, Yilei Shi, Xiao Xiang Zhu. Deep learning-based framework for city-scale rooftop solar potential estimation by considering roof superstructures. Applied Energy,2024.

## Why This Matters
Demonstrates ability to reproduce and adapt research-level code.

Highlights skills in data restructuring, debugging segmentation pipelines, and fine-tuning CNNs.

Sets strong foundation for building more domain-specific architectures (like HRNet pipelines).


# Data Ingestion & Storage Guidelines

This directory explains the dataset requirements, acquisition geometry, calibration protocols, and storage structure for both **Track 1 (RGB Baseline)** and **Track 2 (Hyperspectral PUMA)**.

---

## 📌 Dataset Overview

| Track | Modality | Spectral Channels | Spatial Resolution | File Types | Disk Size |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Track 1** | Visible RGB Images | 3 (400–700 nm) | 1080p / 128×128 crops | `.jpg`, `.png` | ~50 MB |
| **Track 2** | ENVI Hyperspectral Cubes | 1,200 (393.77–1014.73 nm) | 1920 × 1920 pixels | `.spe`, `.hdr`, `.figspec*` | ~26.5 GB |

---

## 📂 File Specification for Track 2 (Hyperspectral)

The raw hyperspectral acquisition contains:
1. `dataset1.spe` (8.24 GiB):
   - Raw Digital Numbers (DN) captured by the hyperspectral pushbroom sensor.
   - Dimensions: $1920 \times 1920 \times 1200$ voxels.
   - Interleave: **BIL** (Band Interleaved by Line).
   - Data type: `uint16` (unsigned 16-bit integer).
2. `dataset1.hdr` (8.8 KB):
   - Standard ENVI text header detailing sensor parameters, geometry, and the 1200 central wavelength definitions.
3. `dataset1_Ref.spe` (16.48 GiB):
   - Reference surface reflectance file (`float32`).
4. `dataset1.figspecwhite` and `dataset1.figspecblack`:
   - Calibrated white reference standard (Spectralon panel, ~99% diffuse reflectance) and dark reference current measurements.
5. `dataset1.xlsx`:
   - Sensor radiometric calibration coefficients and wavelength calibration mapping.
6. `dataset1.jpg`:
   - Full-scene co-registered RGB reference photograph used for visual landmark validation.

---

## 🔒 Git Ingestion & Gitignore Policy

Due to GitHub's individual file limit of **100 MB** and total repository recommendations:
- **Raw `.spe` binary cubes and `.figspec*` calibration files are strictly gitignored**.
- **All extracted numerical checkpoints, analytical CSV tables, publication figures, and the Executive PDF Report ARE tracked**.
- To run experiments with the raw hyperspectral sensor data:
  1. Place `dataset1.spe`, `dataset1.hdr`, and reference files in `track2_hyperspectral_puma/` or `data/` or the repository root.
  2. Verify file integrity using:
     ```bash
     python3 track2_hyperspectral_puma/diagnose_dataset.py .
     ```

"""
Data synthesis module:
Expands training split via random spatial crops (80-100%) and horizontal flips.
Applied strictly to training partition to prevent leakage into validation and test sets.
"""

import os
import cv2
import numpy as np


def synthesize_training_crops(src_dir, dst_dir, n_aug=4, crop_range=(0.80, 1.00), seed=42):
    """
    Synthesizes n_aug additional crops per image in src_dir and writes to dst_dir.
    Maintains original resolution by scaling crops back to original (w, h).
    """
    np.random.seed(seed)
    os.makedirs(dst_dir, exist_ok=True)

    classes = [d for d in sorted(os.listdir(src_dir)) if os.path.isdir(os.path.join(src_dir, d))]
    total_generated = 0

    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        dst_cls_dir = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        filenames = [f for f in os.listdir(src_cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))]
        for fname in filenames:
            src_path = os.path.join(src_cls_dir, fname)
            img = cv2.imread(src_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            # Copy original
            cv2.imwrite(os.path.join(dst_cls_dir, f"orig_{fname}"), img)
            total_generated += 1

            # Generate synthetic variations
            for i in range(n_aug):
                crop_ratio = np.random.uniform(crop_range[0], crop_range[1])
                ch, cw = int(h * crop_ratio), int(w * crop_ratio)
                y0 = np.random.randint(0, h - ch + 1)
                x0 = np.random.randint(0, w - cw + 1)

                crop = img[y0:y0 + ch, x0:x0 + cw]
                crop_resized = cv2.resize(crop, (w, h))

                if np.random.rand() > 0.5:
                    crop_resized = cv2.flip(crop_resized, 1)

                out_name = f"aug{i}_{fname}"
                cv2.imwrite(os.path.join(dst_cls_dir, out_name), crop_resized)
                total_generated += 1

    return total_generated

"""
Dataset Diagnostics for Hyperspectral Face Recognition
Zero-dependency script (uses Python standard library only).
Safe for large files: reads only small binary slices without loading full cubes into memory.
"""

import os
import sys
import struct
import math
import json

def diagnose_dataset(base_dir="."):
    print(f"=" * 70)
    print(f"HYPERSPECTRAL DATASET DIAGNOSTIC REPORT")
    print(f"Directory: {os.path.abspath(base_dir)}")
    print(f"=" * 70)
    
    files = os.listdir(base_dir)
    spe_files = [f for f in files if f.endswith(".spe")]
    hdr_files = [f for f in files if f.endswith(".hdr")]
    jpg_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    calib_files = [f for f in files if "figspec" in f.lower() or f.endswith(".xlsx")]
    
    print(f"\n[1] File Inventory:")
    print(f"  Total files in folder: {len(files)}")
    total_size = sum(os.path.getsize(os.path.join(base_dir, f)) for f in files if os.path.isfile(os.path.join(base_dir, f)))
    print(f"  Total folder size: {total_size / (1024**3):.2f} GB ({total_size:,} bytes)")
    for f in sorted(files):
        sz = os.path.getsize(os.path.join(base_dir, f))
        print(f"    - {f:<25} : {sz:>14,d} bytes ({sz / (1024**2):>9.2f} MB)")
        
    # Inspect HDR file
    hdr_info = {}
    if hdr_files:
        hdr_path = os.path.join(base_dir, hdr_files[0])
        print(f"\n[2] ENVI Header Analysis ({hdr_files[0]}):")
        with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for line in lines:
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip().lower()
                val = val.strip().strip("{}")
                if key in ["samples", "lines", "bands", "header offset", "file type", "data type", "interleave", "byte order", "sn"]:
                    hdr_info[key] = val
                    print(f"    {key:<18}: {val}")
            if "wavelength =" in line.lower() or "wavelength=" in line.lower():
                pass
                
        # Parse wavelengths
        full_text = "".join(lines)
        if "wavelength" in full_text.lower():
            start_idx = full_text.lower().find("wavelength")
            sub = full_text[start_idx:]
            brace_start = sub.find("{")
            brace_end = sub.find("}")
            if brace_start != -1 and brace_end != -1:
                wl_str = sub[brace_start+1:brace_end]
                wl_vals = [float(x.strip()) for x in wl_str.split(",") if x.strip()]
                print(f"    Wavelengths count : {len(wl_vals)} bands")
                print(f"    Wavelength range  : {min(wl_vals):.2f} nm to {max(wl_vals):.2f} nm (Spectral resolution: ~{(max(wl_vals)-min(wl_vals))/len(wl_vals):.2f} nm/band)")
                hdr_info["wavelengths"] = wl_vals

    # Cube validation & byte calculations
    samples = int(hdr_info.get("samples", 1920))
    lines = int(hdr_info.get("lines", 1920))
    bands = int(hdr_info.get("bands", 1200))
    interleave = hdr_info.get("interleave", "bil").lower()
    
    print(f"\n[3] Cube Geometry & Dimensional Integrity:")
    print(f"    Spatial dimensions : {samples} samples (width) x {lines} lines (height)")
    print(f"    Spectral bands     : {bands} channels")
    print(f"    Interleave format  : {interleave.upper()} (Band Interleaved by Line)")
    total_voxels = samples * lines * bands
    print(f"    Total voxels/cube  : {total_voxels:,} elements")
    
    # Analyze Raw vs Ref SPE files
    print(f"\n[4] Hyperspectral Binary Verification:")
    for spe in sorted(spe_files):
        spe_path = os.path.join(base_dir, spe)
        sz = os.path.getsize(spe_path)
        bytes_per_elem = sz / total_voxels
        dtype_name = "uint16 (raw DN)" if bytes_per_elem == 2 else ("float32 (reflectance)" if bytes_per_elem == 4 else "unknown")
        print(f"    File: {spe}")
        print(f"      Size on disk     : {sz:,} bytes ({sz / (1024**3):.2f} GiB)")
        print(f"      Bytes per voxel  : {bytes_per_elem:.2f} bytes -> Inferred dtype: {dtype_name}")
        
        # Line-by-line quick scan for active (non-zero) data
        elem_size = int(bytes_per_elem)
        line_bytes = bands * samples * elem_size
        active_lines = 0
        zero_lines = 0
        nan_lines = 0
        
        # Sample lines
        with open(spe_path, "rb") as f:
            for l in range(0, lines, max(1, lines // 40)): # 40 sample points
                f.seek(l * line_bytes)
                sample_chunk = f.read(min(line_bytes, 8192))
                if elem_size == 4:
                    vals = struct.unpack(f"<{len(sample_chunk)//4}f", sample_chunk)
                    if any(math.isnan(v) for v in vals):
                        nan_lines += 1
                    if all(v == 0.0 for v in vals):
                        zero_lines += 1
                    else:
                        active_lines += 1
                elif elem_size == 2:
                    vals = struct.unpack(f"<{len(sample_chunk)//2}H", sample_chunk)
                    if all(v == 0 for v in vals):
                        zero_lines += 1
                    else:
                        active_lines += 1
                        
        print(f"      Sample line scan : {active_lines} active sampled, {zero_lines} all-zero sampled")
        if zero_lines > 0 and "ref" in spe.lower():
            print(f"      [WARNING] Significant zero-padding detected in {spe}! Likely incomplete calibration export.")

    print(f"\n[5] Memory & Hardware Sizing for Windows 11:")
    cube_ram_f32 = (samples * lines * bands * 4) / (1024**3)
    cube_ram_f64 = cube_ram_f32 * 2
    print(f"    In-memory footprint for 1 full cube:")
    print(f"      - Float32 : {cube_ram_f32:.2f} GiB RAM")
    print(f"      - Float64 : {cube_ram_f64:.2f} GiB RAM")
    print(f"    [STRATEGY REQUIRED]:")
    print(f"      Direct in-memory loading will trigger Out-Of-Memory (OOM) on 16 GB / 32 GB RAM systems.")
    print(f"      Must use `numpy.memmap`, band-wise slicing, or spatial ROI cropping.")
    print(f"=" * 70)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    diagnose_dataset(path)

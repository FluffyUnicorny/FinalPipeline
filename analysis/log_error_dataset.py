import csv
from pathlib import Path
import numpy as np

# ปรับ path ให้ตรงของเธอ
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/error_dataset.csv"

# ----- ใส่ค่าที่ pipeline print ออกมา -----
range_m = 2.392
world_error_m = 2.625        # จาก pipeline
reproj_mean_px = 0.632
reproj_std_px = 0.529
reproj_p95_px = 1.672

# ------------------------------------------

header = [
    "range_m",
    "world_error_m",
    "reproj_mean_px",
    "reproj_std_px",
    "reproj_p95_px",
]

write_header = not OUT.exists()

with open(OUT, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    w.writerow([
        range_m,
        world_error_m,
        reproj_mean_px,
        reproj_std_px,
        reproj_p95_px,
    ])

print("Logged to", OUT)

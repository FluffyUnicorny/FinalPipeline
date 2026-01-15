import pandas as pd

df = pd.read_csv("analysis/error_dataset.csv")

print("\n[Error characterization at fixed range]")
print(f"Range (m)           : {df['range_m'][0]:.3f}")
print(f"World error (m)     : {df['world_error_m'][0]:.3f}")
print(f"Reproj mean (px)    : {df['reproj_mean_px'][0]:.3f}")
print(f"Reproj std (px)     : {df['reproj_std_px'][0]:.3f}")
print(f"Reproj p95 (px)     : {df['reproj_p95_px'][0]:.3f}")

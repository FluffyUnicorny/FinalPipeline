import numpy as np
from pathlib import Path
from colmap_utils.read_write_model import read_model


def compute_reprojection_stats(colmap_sparse_dir):
    """
    Read reprojection error directly from COLMAP points3D.bin
    """

    colmap_sparse_dir = Path(colmap_sparse_dir)

    # อ่าน COLMAP model (.bin)
    _, _, points3D = read_model(str(colmap_sparse_dir), ext=".bin")

    if len(points3D) == 0:
        print("[Reprojection stability]")
        print("No 3D points found.")
        return

    # COLMAP reprojection error (pixel)
    errors = np.array([p.error for p in points3D.values()])

    print()
    print("[Reprojection stability]")
    print(f"mean_px: {errors.mean():.3f}")
    print(f"std_px : {errors.std():.3f}")
    print(f"p95_px : {np.percentile(errors, 95):.3f}")
    print(f"n      : {len(errors)}")

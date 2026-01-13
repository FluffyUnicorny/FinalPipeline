from pathlib import Path
import subprocess
import numpy as np
import os
import shutil
import csv

# =============================
# CONFIG
# =============================
DATASET_NAME = "dataset_real_umbrella"
ROOT = Path(__file__).resolve().parent
SUBSAMPLE_STEP = 9

# =============================
# PATHS
# =============================
IMG_DIR = ROOT / f"data/{DATASET_NAME}/images"
IMG_SUB = ROOT / f"data/{DATASET_NAME}/images_sub"
COLMAP_WS = ROOT / f"data/{DATASET_NAME}/colmap"
SPARSE = COLMAP_WS / "sparse/0"
CAMERA_WORLD_CSV = ROOT / f"data/{DATASET_NAME}/camera_world.csv"

COLMAP_WS.mkdir(parents=True, exist_ok=True)

COLMAP_EXE = r"C:\COLMAP_CPU\bin\colmap.exe"
QT_PLUGIN = r"C:\COLMAP_CPU\plugins\platforms"

from shared.colmap_io import load_colmap_sparse
from refinement.refinement import refine_model

# =============================
# SUBSAMPLE IMAGES
# =============================
def subsample_images():
    if IMG_SUB.exists():
        shutil.rmtree(IMG_SUB)
    IMG_SUB.mkdir()

    imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"])
    use = imgs[::SUBSAMPLE_STEP]
    for p in use:
        shutil.copy(p, IMG_SUB / p.name)

# =============================
# RUN COLMAP
# =============================
def run_colmap():
    db = COLMAP_WS / "database.db"
    sparse = COLMAP_WS / "sparse"
    if db.exists(): db.unlink()
    if sparse.exists(): shutil.rmtree(sparse)
    sparse.mkdir(parents=True)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "windows"
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = QT_PLUGIN

    def run(cmd):
        subprocess.run(cmd, check=True, env=env)

    run([COLMAP_EXE, "feature_extractor", "--database_path", db, "--image_path", IMG_SUB, "--ImageReader.single_camera", "1"])
    run([COLMAP_EXE, "exhaustive_matcher", "--database_path", db])
    run([COLMAP_EXE, "mapper", "--database_path", db, "--image_path", IMG_SUB, "--output_path", sparse])

# =============================
# REFINEMENT WITH LOG
# =============================
def refine_with_log():
    pts, poses = load_colmap_sparse(SPARSE)
    n_before = len(pts)
    pts_r, poses_r = refine_model(points_3d=pts, poses=poses)
    n_after = len(pts_r)
    print()
    print("✅ COLMAP done")
    print(f"▶ Refinement: {n_before} → {n_after} pts")
    return pts_r, poses_r

# =============================
# GEOREFERENCING
# =============================
def load_world_positions():
    with open(CAMERA_WORLD_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    world = []
    for i in range(0, len(rows), SUBSAMPLE_STEP):
        r = rows[i]
        world.append([
            float(r["E"]),
            float(r["N"]),
            -float(r["H"])  # 🔥 แค่ flip sign ของ H
        ])
    return np.array(world)

def similarity(X, Y):
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = np.trace(np.diag(D)) / np.sum(Xc ** 2)
    t = muY - s * R @ muX
    return s, R, t

def georef(pts, poses):
    world = load_world_positions()
    cam_positions = np.array([p["t"] for p in poses.values()])

    # 🔥 flip Z ของ cam_positions ก่อน similarity transform
    cam_positions[:, 2] *= -1

    n = min(len(cam_positions), len(world))
    X = cam_positions[:n]
    Y = world[:n]
    if n < 3:
        return pts, poses
    s, R, t = similarity(X, Y)
    pts_w = (s * (R @ pts.T)).T + t
    poses_w = {k: {"t": (s * (R @ v["t"] * np.array([1,1,-1]))) + t, "R": R @ v["R"]} for k, v in poses.items()}  # flip Z per pose
    return pts_w, poses_w

# =============================
# ERROR
# =============================
def error_vs_real(pts, real_pos):
    est = pts.mean(0)
    err = np.linalg.norm(est - real_pos)
    print(" Real      :", real_pos)
    print("Estimated :", est)
    print(f"Error     : {err:.3f} m")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    subsample_images()
    run_colmap()
    pts, poses = refine_with_log()
    pts_w, poses_w = georef(pts, poses)

    REAL_POS = np.array([100.5653152, 13.84682465, -5.409000397])
    # real_box (1) 100.5653152, 13.84682465, -5.409000397
    # real_umbrella (1) 100.5658646, 13.84652138, -6.760000229
    error_vs_real(pts_w, REAL_POS)

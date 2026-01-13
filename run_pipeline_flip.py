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

    imgs = sorted(
        p for p in IMG_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
    )

    for p in imgs[::SUBSAMPLE_STEP]:
        shutil.copy(p, IMG_SUB / p.name)

# =============================
# RUN COLMAP
# =============================
def run_colmap():
    db = COLMAP_WS / "database.db"
    sparse = COLMAP_WS / "sparse"

    if db.exists():
        db.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "windows"
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = QT_PLUGIN

    def run(cmd):
        subprocess.run(cmd, check=True, env=env)

    run([
        COLMAP_EXE, "feature_extractor",
        "--database_path", db,
        "--image_path", IMG_SUB,
        "--ImageReader.single_camera", "1"
    ])

    run([
        COLMAP_EXE, "exhaustive_matcher",
        "--database_path", db
    ])

    run([
        COLMAP_EXE, "mapper",
        "--database_path", db,
        "--image_path", IMG_SUB,
        "--output_path", sparse
    ])

# =============================
# REFINEMENT
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
# LOAD WORLD POSITIONS (FIX H SIGN)
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
            -float(r["H"])  # 🔥 FIX: world Z must be UP
        ])

    return np.array(world)

# =============================
# SIMILARITY TRANSFORM
# =============================
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

# =============================
# GEOREFERENCING (FINAL)
# =============================
def georef(pts, poses):
    world_all = load_world_positions()
    pose_items = list(poses.items())
    n = min(len(pose_items), len(world_all))

    if n < 3:
        print("❌ Cannot georef: < 3 cameras")
        return pts, poses

    cam_positions = []
    world = []

    for i in range(n):
        cam_positions.append(pose_items[i][1]["t"])
        world.append(world_all[i])

    cam_positions = np.array(cam_positions)
    world = np.array(world)

    # FIX AXIS: COLMAP Z down → Z up
    cam_positions[:, 2] *= -1

    s, R, t = similarity(cam_positions, world)

    pts_w = (s * (R @ pts.T)).T + t

    poses_w = {}
    for k, v in poses.items():
        tc = v["t"].copy()
        tc[2] *= -1
        poses_w[k] = {
            "t": (s * (R @ tc)) + t,
            "R": R @ v["R"]
        }

    print(f"🌍 Georef using {n} cameras")

    return pts_w, poses_w

# =============================
# ERROR (CAMERA CENTROID)
# =============================
def error_vs_real(poses_w, real_pos):
    cams = np.array([p["t"] for p in poses_w.values()])
    est = cams.mean(0)
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
    error_vs_real(poses_w, REAL_POS)

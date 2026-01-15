import numpy as np

def compute_camera_scene_distance(poses_w, pts_w):
    """
    poses_w : dict { image_name: { "t": (3,) } }
    pts_w   : (N,3) array
    """

    cam_positions = np.array([p["t"] for p in poses_w.values()])
    scene_center = pts_w.mean(axis=0)

    dists = np.linalg.norm(cam_positions - scene_center, axis=1)

    return {
        "mean": dists.mean(),
        "std": dists.std(),
        "min": dists.min(),
        "max": dists.max(),
    }

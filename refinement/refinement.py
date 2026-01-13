import numpy as np

def remove_outliers_statistical(points, z_thresh=3.0):
    centroid = np.mean(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)

    mean = np.mean(dists)
    std = np.std(dists)

    mask = dists < mean + z_thresh * std
    return points[mask], mask


def refine_model(points_3d, poses):
    """
    Minimal refinement:
    - statistical outlier removal only
    """

    refined_points = points_3d.copy()
    refined_poses = poses

    refined_points, _ = remove_outliers_statistical(
        refined_points, z_thresh=3.0
    )

    return refined_points, refined_poses

import numpy as np

def compute_error(estimated, real):
    """
    Compute Euclidean distance between estimated point and ground truth
    """
    err = np.linalg.norm(estimated - real)
    return err

if __name__ == "__main__":
    # --- ตัวอย่างค่าพิกัด ---
    gt_pos        = np.array([100.5653152, 13.84682465, -5.409000397])  # Ground truth
    # real_box (1) 100.5653152, 13.84682465, -5.409000397
    # real_umbrella (1) 100.5658646, 13.84652138, -6.760000229
    estimated_pos = np.array([100.32669879, 14.09244535, 5.75836742])  # Pipeline
    tracked_pos   = np.array([100.5676553, 13.8475860, 5.1])  # Camera tracked

    # --- คำนวณ error ---
    error_estimated = compute_error(estimated_pos, gt_pos)
    error_tracked   = compute_error(tracked_pos, gt_pos)

    # --- แสดงผล ---
    print()
    print("Ground truth :", gt_pos)
    print("Estimated    :", estimated_pos, f"Error: {error_estimated:.3f} m")
    print("Tracked      :", tracked_pos,   f"Error: {error_tracked:.3f} m")

    # --- เปรียบเทียบว่าอันไหนดีกว่า ---
    if error_estimated < error_tracked:
        print("\n=> Estimated (pipeline) ใกล้ GT กว่ากล้อง track")
    elif error_estimated > error_tracked:
        print("\n=> Camera tracked ใกล้ GT กว่า pipeline")
    else:
        print("\n=> ทั้งสองใกล้ GT เท่ากัน")

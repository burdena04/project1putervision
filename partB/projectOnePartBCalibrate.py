#!/usr/bin/env python3
# projectOnePartBCalibrate_min.py
# Minimal, robust calibration script (no fancy formatting)

import cv2 as cv
import numpy as np
import glob, argparse, os, json
from pathlib import Path

def detect(gray, pattern_size, use_sb):
    if use_sb:
        try:
            ok, corners = cv.findChessboardCornersSB(gray, pattern_size, flags=cv.CALIB_CB_EXHAUSTIVE)
            if ok:
                return True, corners.astype(np.float32)
        except Exception:
            pass
    ok, corners = cv.findChessboardCorners(
        gray, pattern_size,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    )
    if ok:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    return ok, corners

def rms_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    errs = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv.projectPoints(objp, rvec, tvec, K, dist)
        proj = proj.reshape(-1,2); imgp = imgp.reshape(-1,2)
        err = np.sqrt(np.mean(np.sum((proj - imgp)**2, axis=1)))
        errs.append(float(err))
    return errs, (float(np.mean(errs)) if errs else float("inf"))

def main():
    ap = argparse.ArgumentParser(description="Calibrate camera using chessboard images")
    ap.add_argument("--imgs", default="calib_images/*.png", help="glob pattern for images")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column (vertical)")
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row (horizontal)")
    ap.add_argument("--square_size", type=float, default=0.024, help="square edge length (meters or mm)")
    ap.add_argument("--out", default="calib_out", help="output directory")
    ap.add_argument("--use_sb", action="store_true", help="prefer SB detector")
    ap.add_argument("--preview", action="store_true", help="show quick undistort previews")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    pattern_size = (args.cols, args.rows)

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints, imgpoints, used_files = [], [], []
    images = sorted(glob.glob(args.imgs))
    if not images:
        raise SystemExit("No images matched " + args.imgs)

    for f in images:
        img = cv.imread(f, cv.IMREAD_COLOR)
        if img is None:
            print("[skip] unreadable:", f)
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ok, corners = detect(gray, pattern_size, args.use_sb)
        if not ok:
            print("[skip] pattern not found:", f)
            continue
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used_files.append(f)

    if len(objpoints) < 8:
        raise SystemExit("Too few valid images (" + str(len(objpoints)) + "). Capture 15–30 varied views.")

    h, w = cv.imread(used_files[0]).shape[:2]
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None, flags=cv.CALIB_RATIONAL_MODEL
    )

    per_img, mean_err = rms_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    # save outputs
    np.savez(os.path.join(args.out, "intrinsics.npz"), K=K, dist=dist, image_size=(w,h))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({
            "image_size": [w, h],
            "K": K.tolist(),
            "dist_coeffs": dist.reshape(-1).tolist(),
            "mean_reprojection_error_px": mean_err,
            "per_image_errors_px": {Path(p).name: e for p, e in zip(used_files, per_img)}
        }, f, indent=2)

    # human-readable txt
    def fmt(x): return "{: .6f}".format(x)
    with open(os.path.join(args.out, "intrinsics.txt"), "w") as f:
        f.write("Camera matrix K:\n")
        f.write(np.array2string(K, formatter={'float_kind': fmt}))
        f.write("\n\ndistortion [k1 k2 p1 p2 k3 k4 k5 k6]:\n")
        f.write(np.array2string(dist.reshape(-1), formatter={'float_kind': fmt}))
        f.write("\n\nMean reprojection error (px): " + "{:.6f}".format(mean_err) + "\n")

    print("\n== Calibration results ==")
    print("K =\n", K)
    print("dist =", dist.reshape(-1))
    print("Mean reprojection error:", round(mean_err, 6), "px")
    print("Saved:", os.path.join(args.out, "intrinsics.npz"), "and report.json/intrinsics.txt")

    if args.preview:
        newK, _ = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1.0, (w,h))
        for f in used_files[:min(4, len(used_files))]:
            img = cv.imread(f)
            und = cv.undistort(img, K, dist, None, newK)
            cv.imshow("orig | undistorted", np.hstack([img, und]))
            cv.waitKey(700)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()

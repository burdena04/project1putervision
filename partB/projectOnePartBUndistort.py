import cv2 as cv
import numpy as np
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Undistort an image with optional crop/save")
    ap.add_argument("--params", default="calib_out/intrinsics.npz", help="npz with K, dist, image_size")
    ap.add_argument("--img", required=True, help="image to undistort")
    ap.add_argument("--alpha", type=float, default=0.0, help="0=crop to valid, 1=keep all (default 0.0)")
    ap.add_argument("--save", help="optional output path to save undistorted (cropped) image")
    args = ap.parse_args()

    data = np.load(args.params)
    K, dist = data["K"], data["dist"]

    img = cv.imread(args.img, cv.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read --img '{args.img}'")
    h, w = img.shape[:2]

    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), args.alpha, (w, h))
    und = cv.undistort(img, K, dist, None, newK)

    # crop to valid region if alpha < 1
    x, y, ww, hh = roi
    if ww > 0 and hh > 0:
        und_cropped = und[y:y+hh, x:x+ww]
    else:
        und_cropped = und

    cv.imshow("original", img)
    cv.imshow("undistorted", und_cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(args.save, und_cropped)
        print(f"[saved] {args.save}")

if __name__ == "__main__":
    main()

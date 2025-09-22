# batch_undistort.py
import cv2 as cv, numpy as np, glob, os, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--params", default="calib_out/intrinsics.npz")
ap.add_argument("--in_glob", default="calib_images/*.png")
ap.add_argument("--out_dir", default="calib_out/undistorted")
ap.add_argument("--alpha", type=float, default=0.0)
args = ap.parse_args()

data = np.load(args.params); K, dist = data["K"], data["dist"]
files = sorted(glob.glob(args.in_glob))
Path(args.out_dir).mkdir(parents=True, exist_ok=True)

for f in files:
    img = cv.imread(f); h, w = img.shape[:2]
    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), args.alpha, (w,h))
    und = cv.undistort(img, K, dist, None, newK)
    x,y,ww,hh = roi
    if ww>0 and hh>0: und = und[y:y+hh, x:x+ww]
    out = os.path.join(args.out_dir, Path(f).stem + "_undist.png")
    cv.imwrite(out, und); print("[saved]", out)

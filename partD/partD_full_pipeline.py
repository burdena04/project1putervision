#!/usr/bin/env python3
"""Part D: Integrated capture -> undistort -> detect -> pose pipeline."""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import pupil_apriltags as apriltag

# Allow importing helpers from Part C without turning the repo into a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from partC.partC_localize_apriltag import (
    load_world,
    detect_and_pose,
    draw_dets,
    draw_axes,
    open_cam,
    blend_pose,
    R_to_quat,
)


class PoseLogger:
    """CSV logger to capture robustness-test results (logs raw per-detection pose)."""

    def __init__(self, path, scenario):
        self.scenario = scenario
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fh = p.open("w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow([
            "timestamp",
            "scenario",
            "frame_mean",
            "tag_count",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
        ])

    def write(self, frame_mean, tag_count, pose, quat):
        self._writer.writerow([
            time.time(),
            self.scenario,
            frame_mean,
            tag_count,
            pose[0],
            pose[1],
            pose[2],
            quat[0],
            quat[1],
            quat[2],
            quat[3],
        ])
        self._fh.flush()

    def close(self):
        self._fh.close()


def undistort_image(img, K, dist, alpha):
    """Return undistorted image plus the rectified intrinsics."""
    h, w = img.shape[:2]
    newK, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    und = cv.undistort(img, K, dist, None, newK)
    return und, newK


def build_detector(args):
    return apriltag.Detector(
        families=args.family,
        nthreads=args.threads,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=not args.no_refine,
    )


def run_image_mode(args, K, dist, tag_size, world):
    img = cv.imread(args.img, cv.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.img}")

    und, newK = undistort_image(img, K, dist, args.alpha)
    gray = cv.cvtColor(und, cv.COLOR_BGR2GRAY)
    zero_dist = np.zeros_like(dist)

    detector = build_detector(args)
    R_wc, t_wc, ids, dets = detect_and_pose(
        gray,
        newK,
        zero_dist,
        tag_size,
        world,
        detector,
        scale=args.scale,
        target_tag=args.only_tag,
    )

    if R_wc is None:
        raise SystemExit("No known tags found in undistorted image.")

    vis = und.copy()
    draw_dets(vis, dets, args.scale)
    if args.draw_axes:
        vis = draw_axes(vis, newK, zero_dist, R_wc, t_wc, 0.05)

    cv.imshow("raw", img)
    cv.imshow("undistorted", vis)
    cv.waitKey(0)
    cv.destroyAllWindows()

    qx, qy, qz, qw = R_to_quat(R_wc)
    print("Tag IDs:", ids)
    print("Camera position (m):", t_wc.ravel())
    print("Quaternion (x,y,z,w):", (qx, qy, qz, qw))

    if args.save_undist:
        Path(args.save_undist).parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(args.save_undist, und)
        print(f"Saved undistorted image to {args.save_undist}")


def run_camera_mode(args, K, dist, tag_size, world):
    detector = build_detector(args)
    cap = open_cam(args.cam, args.backend)
    if not cap.isOpened():
        raise SystemExit("Failed to open camera. Try another --cam or --backend.")

    zero_dist = np.zeros_like(dist)
    logger = PoseLogger(args.log, args.scenario) if args.log else None

    print("Press q to quit, space to toggle detection")
    enable_detect = True
    last_R, last_t, last_ids = None, None, []
    pose_alpha = float(np.clip(args.pose_ema, 0.0, 1.0))
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; exiting.")
                break

            und, newK = undistort_image(frame, K, dist, args.alpha)
            gray = cv.cvtColor(und, cv.COLOR_BGR2GRAY)

            vis = und.copy()
            frame_mean = float(gray.mean())
            run_this = enable_detect and (frame_idx % max(1, args.detect_every) == 0)
            dets = []
            raw_pose = None

            if run_this:
                R_wc, t_wc, ids, dets = detect_and_pose(
                    gray,
                    newK,
                    zero_dist,
                    tag_size,
                    world,
                    detector,
                    scale=args.scale,
                    target_tag=args.only_tag,
                )
                if R_wc is not None:
                    raw_pose = (R_wc, t_wc, ids)
                    last_R, last_t = blend_pose(last_R, last_t, R_wc, t_wc, pose_alpha)
                    last_ids = ids

            if dets:
                draw_dets(vis, dets, args.scale)

            if last_t is not None and args.draw_axes:
                vis = draw_axes(vis, newK, zero_dist, last_R, last_t, 0.05)

            status = (
                f"detect={'on' if enable_detect else 'off'} "
                f"mean={frame_mean:.1f} tags={len(last_ids)}"
            )
            cv.putText(vis, status, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if last_t is not None:
                qx, qy, qz, qw = R_to_quat(last_R)
                pose_str = f"x:{last_t[0,0]:.3f} y:{last_t[1,0]:.3f} z:{last_t[2,0]:.3f}"
                cv.putText(vis, pose_str, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if raw_pose and logger:
                raw_R, raw_t, raw_ids = raw_pose
                q_raw = R_to_quat(raw_R)
                logger.write(
                    frame_mean,
                    len(raw_ids),
                    (raw_t[0, 0], raw_t[1, 0], raw_t[2, 0]),
                    q_raw,
                )

            cv.imshow("partD pipeline", vis)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                enable_detect = not enable_detect

            frame_idx += 1
    finally:
        if logger:
            logger.close()
        cap.release()
        cv.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Integrated Part D pipeline: capture -> undistort -> detect -> pose"
    )
    ap.add_argument("--params", required=True, help="Calibration npz containing K/dist")
    ap.add_argument("--world", required=True, help="World tag configuration JSON")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--img", help="Process a single image file")
    g.add_argument("--cam", type=int, help="Camera index for live mode")

    ap.add_argument("--alpha", type=float, default=0.0, help="Undistort alpha (0 valid crop, 1 keep all)")
    ap.add_argument("--scale", type=float, default=0.6, help="Downscale factor for AprilTag detector")
    ap.add_argument("--detect-every", type=int, default=2, help="Run detection every N frames in live mode")
    ap.add_argument("--only-tag", type=int, help="Restrict pose estimation to a single tag ID")
    ap.add_argument("--pose-ema", type=float, default=1.0, help="EMA weight for pose smoothing (live)")
    ap.add_argument("--draw-axes", action="store_true", help="Overlay pose axes on undistorted view")
    ap.add_argument("--save-undist", help="Path to save undistorted image in image mode")
    ap.add_argument("--log", help="CSV file to log detections (live mode)")
    ap.add_argument("--scenario", default="baseline", help="Label describing test condition for the log")
    ap.add_argument("--family", default="tag36h11", help="AprilTag family")
    ap.add_argument("--threads", type=int, default=4, help="Detector worker threads")
    ap.add_argument("--backend", choices=["msmf", "dshow", "auto"], default="auto", help="Preferred camera backend")
    ap.add_argument("--no-refine", action="store_true", help="Disable AprilTag edge refinement (useful for stress tests)")
    return ap.parse_args()


def main():
    args = parse_args()

    data = np.load(args.params)
    K, dist = data["K"], data["dist"]
    tag_size, world = load_world(args.world)

    if args.img:
        run_image_mode(args, K, dist, tag_size, world)
    else:
        run_camera_mode(args, K, dist, tag_size, world)


if __name__ == "__main__":
    main()

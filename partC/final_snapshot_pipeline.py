import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np


def quat_to_R(qx, qy, qz, qw):
    qx, qy, qz, qw = map(float, (qx, qy, qz, qw))
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw),     1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx*qx + qy*qy)],
    ], dtype=float)


def load_pose(args):
    pose_path = Path(args.pose_json)
    pose = json.loads(pose_path.read_text())

    t = np.array(pose['translation_m'], dtype=float)
    qx, qy, qz, qw = pose['quaternion_xyzw']
    R = quat_to_R(qx, qy, qz, qw)

    cam_matrix = np.array(pose['cam_matrix'], dtype=float) if 'cam_matrix' in pose else None
    dist = None
    if 'dist_coeffs' in pose:
        dist = np.array(pose['dist_coeffs'], dtype=float)
        if dist.ndim == 1:
            dist = dist.reshape(-1, 1)

    if cam_matrix is None or dist is None:
        if args.params is None:
            if cam_matrix is None:
                raise ValueError('Camera matrix not found; provide --params or ensure pose JSON includes cam_matrix')
        else:
            data = np.load(args.params)
            if cam_matrix is None and 'K' in data:
                cam_matrix = np.array(data['K'], dtype=float)
            if dist is None and 'dist' in data:
                dist = np.array(data['dist'], dtype=float)
                if dist.ndim == 1:
                    dist = dist.reshape(-1, 1)

    if cam_matrix is None:
        raise ValueError('Camera matrix unavailable; supply --params')
    if dist is None:
        dist = np.zeros((5, 1), dtype=float)

    frame_undistorted = bool(pose.get('frame_undistorted', False))

    return R, t, cam_matrix, dist, frame_undistorted


def intersect_wall(ray_origin, ray_dir):
    z0 = float(ray_origin[2])
    dz = float(ray_dir[2])
    if abs(dz) < 1e-8:
        raise ValueError('Ray parallel to wall plane (z=0)')
    lam = -z0 / dz
    return ray_origin + lam * ray_dir


def collect_points(img, prompt):
    disp = img.copy()
    cv.putText(disp, prompt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv.circle(disp, (x, y), 4, (0, 0, 255), -1)
        elif event == cv.EVENT_RBUTTONDOWN:
            pts.clear()
            disp[:] = img
            cv.putText(disp, prompt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv.namedWindow('snapshot')
    cv.setMouseCallback('snapshot', on_mouse)
    while True:
        cv.imshow('snapshot', disp)
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            cv.destroyAllWindows()
            raise SystemExit('Cancelled')
        if key in (13, 10) and len(pts) >= 2:
            break
    cv.destroyAllWindows()
    return pts[:2]


def pixels_to_world(points, R, t, cam_matrix):
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    world_pts = []
    for u, v in points:
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray_cam = np.array([x, y, 1.0], dtype=float)
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = R @ ray_cam
        world_pts.append(intersect_wall(t, ray_world))
    return world_pts


def draw_overlay(img, pt1, pt2, meters, gt=None):
    out = img.copy()
    p1 = tuple(map(int, pt1))
    p2 = tuple(map(int, pt2))
    cv.line(out, p1, p2, (0, 0, 255), 2)
    cv.circle(out, p1, 5, (255, 0, 0), -1)
    cv.circle(out, p2, 5, (255, 0, 0), -1)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    label = f'{meters*100:.1f} cm'
    if gt is not None:
        label += f' (gt {gt*100:.1f} cm)'
    cv.putText(out, label, mid, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description='Measure object length from saved snapshot pose')
    ap.add_argument('--pose-json', required=True, help='snapshot metadata JSON produced by partC_localize_apriltag')
    ap.add_argument('--img', required=True, help='corresponding snapshot image (PNG/JPG)')
    ap.add_argument('--params', help='intrinsics npz (fallback if pose JSON lacks cam_matrix/dist coefficients)')
    ap.add_argument('--gt', type=float, help='ground truth length in meters for error reporting')
    ap.add_argument('--out-image', help='path to save annotated overlay')
    ap.add_argument('--points', nargs=4, type=float, metavar=('u1', 'v1', 'u2', 'v2'), help='predefined pixel coordinates')
    ap.add_argument('--skip-undistort', action='store_true', help='treat frame as already undistorted regardless of metadata')
    ap.add_argument('--force-undistort', action='store_true', help='always undistort using intrinsics/distortion')
    args = ap.parse_args()

    R, t, cam_matrix, dist, frame_undistorted = load_pose(args)

    img = cv.imread(args.img, cv.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'Could not load image: {args.img}')

    do_undistort = not frame_undistorted
    if args.skip_undistort:
        do_undistort = False
    if args.force_undistort:
        do_undistort = True

    if do_undistort:
        und = cv.undistort(img, cam_matrix, dist)
        cam_used = cam_matrix
    else:
        und = img.copy()
        cam_used = cam_matrix

    if args.points:
        pts_px = [(args.points[0], args.points[1]), (args.points[2], args.points[3])]
    else:
        pts_px = collect_points(und, 'click two endpoints (left), right-click clears, press Enter when done')

    pts_world = pixels_to_world(pts_px, R, t, cam_used)
    meters = float(np.linalg.norm(pts_world[0] - pts_world[1]))
    print(f'Measured: {meters:.4f} m')
    if args.gt is not None:
        err = meters - args.gt
        pct = (err / args.gt) * 100.0 if args.gt != 0 else float('inf')
        print(f'Ground truth: {args.gt:.4f} m -> error {err:.4f} m ({pct:.2f}%)')

    overlay = draw_overlay(und, pts_px[0], pts_px[1], meters, args.gt)
    cv.imshow('measurement', overlay)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if args.out_image:
        out_path = Path(args.out_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out_path), overlay)
        print(f'Saved overlay to {out_path}')


if __name__ == '__main__':
    main()

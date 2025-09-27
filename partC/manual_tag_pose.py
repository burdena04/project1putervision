"""Manual AprilTag pose estimation via SolvePnP."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2 as cv
import numpy as np

from partC.partC_localize_apriltag import load_world


def load_intrinsics(path):
    data = np.load(path)
    K = data['K'].astype(np.float64)
    dist = data['dist'].astype(np.float64)
    return K, dist


def r_to_quat(R):
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    return q[0], q[1], q[2], q[3]


def annotate(base_img, clicks, order):
    img = base_img.copy()
    for tid, xy in clicks:
        cv.circle(img, (int(xy[0]), int(xy[1])), 4, (0, 0, 255), -1)
        cv.putText(img, f'id {tid}', (int(xy[0]) + 6, int(xy[1]) - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    remaining = [tid for tid in order if tid not in [t for t, _ in clicks]]
    if remaining:
        prompt = f'Click center of tag ID {remaining[0]} (left); right-click undoes; Enter solves'
    else:
        prompt = 'All tags marked. Press Enter to solve (Esc to cancel).'
    cv.putText(img, prompt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               (0, 255, 0) if not remaining else (255, 255, 0), 2)
    return img


def collect_points(image, order):
    clicks = []
    disp = annotate(image, clicks, order)
    state = {'disp': disp, 'order': order, 'clicks': clicks}

    def on_mouse(event, x, y, *_):
        if event == cv.EVENT_LBUTTONDOWN:
            remaining = [tid for tid in state['order'] if tid not in [t for t, _ in state['clicks']]]
            if not remaining:
                return
            tid = remaining[0]
            state['clicks'].append((tid, (float(x), float(y))))
            state['disp'] = annotate(image, state['clicks'], state['order'])
        elif event == cv.EVENT_RBUTTONDOWN and state['clicks']:
            state['clicks'].pop()
            state['disp'] = annotate(image, state['clicks'], state['order'])

    cv.namedWindow('manual_pnp', cv.WINDOW_NORMAL)
    cv.setMouseCallback('manual_pnp', on_mouse)

    while True:
        cv.imshow('manual_pnp', state['disp'])
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            cv.destroyAllWindows()
            raise SystemExit('Cancelled')
        if key in (13, 10):
            if len(state['clicks']) == len(order):
                cv.destroyAllWindows()
                return state['clicks']
            print('Need more tags marked before solving.')


def main():
    ap = argparse.ArgumentParser(description='Manual AprilTag pose via solvePnP (tag centers).')
    ap.add_argument('--params', required=True, help='Camera intrinsics npz (K, dist)')
    ap.add_argument('--world', required=True, help='World tag JSON with tag centers/orientation')
    ap.add_argument('--img', required=True, help='Image containing the AprilTags')
    ap.add_argument('--tags', type=int, nargs='+', help='Explicit list of tag IDs to use (default: all in world file)')
    ap.add_argument('--undistort', action='store_true', help='Undistort the image before clicking (zeros distortion)')
    ap.add_argument('--undistort-alpha', type=float, default=0.0, help='Alpha for getOptimalNewCameraMatrix when --undistort is used')
    ap.add_argument('--out-json', help='Write pose to JSON file in snapshot format')
    ap.add_argument('--out-image', help='Write annotated preview image with axes overlay')
    ap.add_argument('--axis-length', type=float, default=0.05, help='Axis length in meters for overlay')
    args = ap.parse_args()

    K, dist = load_intrinsics(args.params)
    _, world = load_world(args.world)

    if args.tags is None:
        tag_order = [tid for tid in sorted(world.keys())]
    else:
        tag_order = [tid for tid in args.tags if tid in world]
    if len(tag_order) < 3:
        raise SystemExit('Need at least 3 tag IDs with known world coordinates for solvePnP.')

    img = cv.imread(args.img, cv.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'Could not read image: {args.img}')

    cam_K = K.copy()
    dist_coeffs = dist.copy()
    display_img = img
    if args.undistort:
        h, w = img.shape[:2]
        newK, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), args.undistort_alpha, (w, h))
        display_img = cv.undistort(img, K, dist, None, newK)
        cam_K = newK
        dist_coeffs = np.zeros_like(dist)

    clicks = collect_points(display_img, tag_order)

    object_pts = []
    image_pts = []
    used_ids = []
    for tid, (x, y) in clicks:
        object_pts.append(world[tid][1].reshape(3))
        image_pts.append([x, y])
        used_ids.append(tid)

    object_pts = np.array(object_pts, dtype=float).reshape(-1, 3)
    image_pts = np.array(image_pts, dtype=float).reshape(-1, 1, 2)

    success, rvec, tvec = cv.solvePnP(object_pts, image_pts, cam_K, dist_coeffs)
    if not success:
        raise SystemExit('solvePnP failed. Check selections and tag coverage.')

    R_co, _ = cv.Rodrigues(rvec)
    t_co = tvec.reshape(3, 1)
    R_wc = R_co.T
    t_wc = -R_wc @ t_co

    print('Used tag IDs:', used_ids)
    print('Camera position (m):', t_wc.ravel())
    qx, qy, qz, qw = r_to_quat(R_wc)
    print('Quaternion (x,y,z,w):', (qx, qy, qz, qw))

    axis_len = float(args.axis_length)
    pts3d = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    proj, _ = cv.projectPoints(pts3d, rvec, tvec, cam_K, dist_coeffs)
    proj = proj.reshape(-1, 2).astype(int)
    annotated = display_img.copy()
    for tid, (x, y) in clicks:
        cv.circle(annotated, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv.putText(annotated, f'id {tid}', (int(x) + 6, int(y) - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    origin = tuple(proj[0])
    cv.line(annotated, origin, tuple(proj[1]), (0, 0, 255), 2)
    cv.line(annotated, origin, tuple(proj[2]), (0, 255, 0), 2)
    cv.line(annotated, origin, tuple(proj[3]), (255, 0, 0), 2)
    cv.putText(annotated, f'x:{t_wc[0,0]:.3f} y:{t_wc[1,0]:.3f} z:{t_wc[2,0]:.3f} m',
               (10, annotated.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow('manual solvePnP result', annotated)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if args.out_image:
        cv.imwrite(args.out_image, annotated)
        print('Wrote annotated image to', args.out_image)

    if args.out_json:
        pose_info = {
            'timestamp': datetime.now().isoformat(),
            'translation_m': [float(t_wc[0,0]), float(t_wc[1,0]), float(t_wc[2,0])],
            'quaternion_xyzw': [float(qx), float(qy), float(qz), float(qw)],
            'used_tags': used_ids,
            'params_file': args.params,
            'world_file': args.world,
            'image_file': args.img,
            'undistorted_image': bool(args.undistort),
            'cam_matrix': cam_K.tolist(),
            'solver': 'manual_solvepnp'
        }
        Path(args.out_json).write_text(json.dumps(pose_info, indent=2))
        print('Wrote pose JSON to', args.out_json)

if __name__ == '__main__':
    main()

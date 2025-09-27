import argparse, json
from pathlib import Path

import numpy as np
import cv2 as cv


def quat_to_R(qx, qy, qz, qw):
    qx, qy, qz, qw = map(float, (qx, qy, qz, qw))
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=float)


def intersect_wall(ray_origin, ray_dir):
    z0 = ray_origin[2]
    dz = ray_dir[2]
    if abs(dz) < 1e-8:
        raise ValueError("Ray parallel to wall plane (z=0)")
    lam = -z0 / dz
    return ray_origin + lam * ray_dir


def main():
    ap = argparse.ArgumentParser(description="Measure distance on the wall plane from clicked pixels")
    ap.add_argument('--params', required=True)
    ap.add_argument('--pose-json', required=True, help='snapshot JSON from partC_localize_apriltag --snapshot-dir')
    ap.add_argument('--img', required=True, help='snapshot PNG (undistorted frame)')
    ap.add_argument('--gt', type=float, help='ground truth length in meters (for error report)')
    ap.add_argument('--points', nargs=4, type=float, metavar=('u1','v1','u2','v2'), help='predefined pixel coordinates')
    ap.add_argument('--skip-undistort', action='store_true', help='Assume the provided frame is already undistorted')
    ap.add_argument('--force-undistort', action='store_true', help='Always undistort the input frame even if the snapshot metadata says it is undistorted')
    ap.add_argument('--update-world', action='store_true', help='Scale world JSON using measured vs ground truth length')
    args = ap.parse_args()

    data = np.load(args.params)
    K, dist = data['K'], data['dist']

    pose_meta = json.loads(Path(args.pose_json).read_text())
    t = np.array(pose_meta['translation_m'], dtype=float)
    qx,qy,qz,qw = pose_meta['quaternion_xyzw']
    R = quat_to_R(qx,qy,qz,qw)

    img = cv.imread(args.img, cv.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'Could not read image: {args.img}')

    cam_K = np.array(pose_meta.get('cam_matrix', K.tolist()), dtype=float)
    frame_undistorted = bool(pose_meta.get('frame_undistorted', False))

    do_undistort = not frame_undistorted
    if args.skip_undistort:
        do_undistort = False
    if args.force_undistort:
        do_undistort = True

    if do_undistort:
        und = cv.undistort(img, K, dist)
        cam_matrix = np.array(K, dtype=float)
    else:
        und = img.copy()
        cam_matrix = cam_K

    fx, fy = cam_matrix[0,0], cam_matrix[1,1]
    cx, cy = cam_matrix[0,2], cam_matrix[1,2]

    points = []
    if args.points:
        points = [(args.points[0], args.points[1]), (args.points[2], args.points[3])]
    else:
        disp = und.copy()
        info = 'click two corners (left button), right-click to reset, press Enter when done'
        print(info)
        cv.putText(disp, info, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        pts = []
        def on_mouse(event,x,y,flags,param):
            if event == cv.EVENT_LBUTTONDOWN:
                pts.append((x,y))
                cv.circle(disp,(x,y),4,(0,0,255),-1)
            elif event == cv.EVENT_RBUTTONDOWN:
                pts.clear()
                disp[:] = und
                cv.putText(disp, info, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv.namedWindow('measure')
        cv.setMouseCallback('measure', on_mouse)
        while True:
            cv.imshow('measure', disp)
            key = cv.waitKey(1) & 0xFF
            if key == 27:
                cv.destroyAllWindows()
                raise SystemExit('Cancelled')
            if key in (13, 10) and len(pts) >= 2:
                points = pts[:2]
                break
        cv.destroyAllWindows()

    pts_world = []
    for (u,v) in points:
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray_cam = np.array([x, y, 1.0], dtype=float)
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = R @ ray_cam
        P = intersect_wall(t, ray_world)
        pts_world.append(P)

    dist_m = float(np.linalg.norm(pts_world[0] - pts_world[1]))
    print(f'Measured distance: {dist_m:.4f} m')
    if args.gt:
        err = dist_m - args.gt
        pct = (err / args.gt) * 100.0
        print(f'Ground truth: {args.gt:.4f} m -> error {err*100:.2f} cm ({pct:.2f}%)')

        if args.update_world:
            if dist_m <= 1e-9:
                print('[update_world] measured distance too small; skipped')
            else:
                scale = args.gt / dist_m
                world_file = pose_meta.get('world_file')
                if not world_file:
                    print('[update_world] snapshot missing world_file, cannot update')
                else:
                    world_path = Path(world_file)
                    if not world_path.is_absolute():
                        world_path = (Path(args.pose_json).parent / world_path).resolve()
                    try:
                        data = json.loads(world_path.read_text())
                    except Exception as exc:
                        print(f'[update_world] failed to read {world_path}: {exc}')
                    else:
                        try:
                            data['tag_size'] = round(float(data['tag_size']) * scale, 6)
                            for tag in data['tags'].values():
                                for key in ('x','y','z'):
                                    tag[key] = round(float(tag[key]) * scale, 6)
                        except Exception as exc:
                            print(f'[update_world] failed to scale world data: {exc}')
                        else:
                            world_path.write_text(json.dumps(data, indent=2))
                            print(f'[update_world] scaled {world_path} by {scale:.6f}')

    out = und.copy()
    p1 = tuple(map(int, points[0]))
    p2 = tuple(map(int, points[1]))
    cv.line(out, p1, p2, (0,0,255), 2)
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    cv.putText(out, f'{dist_m*100:.1f} cm', mid, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv.imshow('measurement', out)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

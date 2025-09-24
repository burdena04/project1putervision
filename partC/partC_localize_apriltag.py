#!/usr/bin/env python3
# AprilTag localization (Part C) — Windows-stable + smooth preview
# deps: pip install pupil-apriltags opencv-python numpy

import argparse, json, math, csv, time
from pathlib import Path
import numpy as np
import cv2 as cv
import pupil_apriltags as apriltag
from datetime import datetime

# ---------- math helpers ----------
def rotz(t):
    c,s=np.cos(t),np.sin(t); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
def se3_inv(R,t): Rt=R.T; return Rt, -Rt@t
def R_to_quat(R):
    K=np.array([
        [R[0,0]-R[1,1]-R[2,2], R[1,0]+R[0,1],        R[2,0]+R[0,2],        R[1,2]-R[2,1]],
        [R[1,0]+R[0,1],        R[1,1]-R[0,0]-R[2,2], R[2,1]+R[1,2],        R[2,0]-R[0,2]],
        [R[2,0]+R[0,2],        R[2,1]+R[1,2],        R[2,2]-R[0,0]-R[1,1], R[0,1]-R[1,0]],
        [R[1,2]-R[2,1],        R[2,0]-R[0,2],        R[0,1]-R[1,0],        R[0,0]+R[1,1]+R[2,2]]
    ],float)/3.0
    w,V=np.linalg.eigh(K); q=V[:,np.argmax(w)]; return q[0],q[1],q[2],q[3]
def average_rotations(Rs):
    M=np.zeros((3,3)); 
    for R in Rs: M+=R
    U,_,Vt=np.linalg.svd(M); Rm=U@Vt
    if np.linalg.det(Rm)<0: U[:,-1]*=-1; Rm=U@Vt
    return Rm

def quat_to_R(qx,qy,qz,qw):
    qx,qy,qz,qw = float(qx),float(qy),float(qz),float(qw)
    n = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
    if n == 0.0: return np.eye(3)
    qx,qy,qz,qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], float)


def blend_pose(prev_R, prev_t, new_R, new_t, alpha):
    beta = float(np.clip(alpha, 0.0, 1.0))
    if prev_R is None or prev_t is None or beta >= 0.999:
        return new_R, new_t
    if beta <= 0.001:
        return prev_R, prev_t
    prev_q = np.array(R_to_quat(prev_R))
    new_q = np.array(R_to_quat(new_R))
    if np.dot(prev_q, new_q) < 0: new_q = -new_q
    q = (1.0-beta)*prev_q + beta*new_q
    q /= np.linalg.norm(q)
    R = quat_to_R(*q)
    t = (1.0-beta)*prev_t + beta*new_t
    return R, t

# ---------- world / drawing ----------
def load_world(path):
    cfg=json.load(open(path,"r"))
    size=float(cfg["tag_size"]); tags={}
    for k,v in cfg["tags"].items():
        tid=int(k)
        x,y,z=float(v["x"]),float(v["y"]),float(v["z"])
        yaw=math.radians(float(v.get("yaw_deg",0.0)))
        tags[tid]=(rotz(yaw), np.array([x,y,z],float).reshape(3,1))
    return size, tags

def draw_axes(img,K,dist,R_wc,t_wc,axis_len=0.05):
    pts=np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]])
    R_cw,t_cw=R_wc.T, -R_wc.T@t_wc
    rvec,_=cv.Rodrigues(R_cw); im,_=cv.projectPoints(pts,rvec,t_cw,K,dist)
    im=im.reshape(-1,2).astype(int); o,x,y,z=im
    cv.line(img,o,x,(0,0,255),2); cv.line(img,o,y,(0,255,0),2); cv.line(img,o,z,(255,0,0),2); return img

def draw_dets(vis,dets,scale):
    s=(1.0/scale) if scale!=1.0 else 1.0
    for d in dets:
        pts=(d.corners*s).astype(int)
        for i in range(4): cv.line(vis,tuple(pts[i]),tuple(pts[(i+1)%4]),(0,255,255),2)
        c=tuple((d.center*s).astype(int)); cv.circle(vis,c,3,(0,0,255),-1)
        cv.putText(vis,str(int(d.tag_id)),c,cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

def has_distortion(dist):
    if dist is None:
        return False
    dist = np.asarray(dist).ravel()
    return dist.size > 0 and np.linalg.norm(dist) > 1e-9

def zero_distortion_like(dist):
    if dist is None:
        return None
    return np.zeros_like(dist)

# ---------- camera ----------
def open_cam(index, backend):
    # force stable backends; set small buffer and 720p
    if backend=="msmf":
        cap=cv.VideoCapture(index, cv.CAP_MSMF)
        if not cap.isOpened(): cap=cv.VideoCapture(index, cv.CAP_DSHOW)
    elif backend=="dshow":
        cap=cv.VideoCapture(index, cv.CAP_DSHOW)
        if not cap.isOpened(): cap=cv.VideoCapture(index, cv.CAP_MSMF)
    else:
        cap=cv.VideoCapture(index, cv.CAP_MSMF)
        if not cap.isOpened(): cap=cv.VideoCapture(index, cv.CAP_DSHOW)
    if not cap.isOpened(): cap=cv.VideoCapture(index)
    # reduce lag and resolution
    try: cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    except: pass
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, 30)
    return cap

# ---------- detection ----------
def detect_and_pose(gray,K,dist,tag_size,world,detector,scale=0.6,target_tag=None):
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    work_gray=gray
    if scale!=1.0:
        work_gray=cv.resize(gray,None,fx=scale,fy=scale,interpolation=cv.INTER_AREA)
        fx,fy,cx,cy=fx*scale,fy*scale,cx*scale,cy*scale
    D=detector.detect(work_gray,estimate_tag_pose=True,camera_params=(fx,fy,cx,cy),tag_size=tag_size)
    Rl,Tl,ids=[],[],[]
    for d in D:
        tid=int(d.tag_id)
        if target_tag is not None and tid != target_tag: continue
        if tid not in world: continue
        R_ct,t_ct=d.pose_R,d.pose_t          # tag in camera
        R_tc,t_tc=se3_inv(R_ct,t_ct)          # camera in tag
        R_wt,t_wt=world[tid]                  # tag in world
        R_wc=R_wt@R_tc; t_wc=R_wt@t_tc+t_wt   # camera in world
        Rl.append(R_wc); Tl.append(t_wc); ids.append(tid)
    if not Rl: return None,None,[],D
    R_wc=average_rotations(Rl); t_wc=np.mean(np.hstack(Tl),axis=1,keepdims=True)
    return R_wc,t_wc,ids,D

def main():
    ap=argparse.ArgumentParser(description="AprilTag-based camera localization (Part C)")
    ap.add_argument("--params", required=True)
    ap.add_argument("--world", required=True)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--img"); g.add_argument("--cam", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--snapshot-dir")
    # NEW knobs:
    ap.add_argument("--backend", choices=["msmf","dshow","auto"], default="dshow")  # dshow is usually least janky
    ap.add_argument("--scale", type=float, default=0.5)                             # downscale for detection (speed)
    ap.add_argument("--detect_every", type=int, default=3, help="run detection every N frames")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--only-tag", type=int, help="Restrict pose estimation to a single tag ID")
    ap.add_argument("--pose-ema", type=float, default=1.0, help="EMA weight for new poses (1.0 disables smoothing)")
    ap.add_argument("--draw-axes", action="store_true", help="Overlay XYZ axes for the camera pose")
    ap.add_argument("--undistort-alpha", type=float, default=0.0, help="Alpha for cv.getOptimalNewCameraMatrix (0=crop,1=keep)")

    ap.add_argument("--threads", type=int, default=4, help="AprilTag detector threads")
    args=ap.parse_args()

    data=np.load(args.params)
    K,dist=data["K"],data["dist"]
    K=np.asarray(K,float)
    dist=np.asarray(dist,float)
    base_K = K.copy()
    cam_K = base_K.copy()
    tag_size,world=load_world(args.world)
    undistort_frames=has_distortion(dist)
    zero_dist=zero_distortion_like(dist) if undistort_frames else dist
    detector=apriltag.Detector(families=args.family,nthreads=args.threads,refine_edges=True)

    if args.img:
        img=cv.imread(args.img,cv.IMREAD_COLOR)
        if img is None: raise SystemExit("Could not read image: "+args.img)
        axes_dist = zero_dist
        if undistort_frames:
            h,w = img.shape[:2]
            newK,_ = cv.getOptimalNewCameraMatrix(base_K,dist,(w,h),args.undistort_alpha,(w,h))
            cam_K = newK
            img=cv.undistort(img,base_K,dist,None,newK)
            axes_dist = zero_dist
        else:
            cam_K = base_K
            axes_dist = dist
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        R_wc,t_wc,ids,dets=detect_and_pose(gray,cam_K,zero_dist if undistort_frames else dist,tag_size,world,detector,scale=args.scale,target_tag=args.only_tag)
        if R_wc is None: raise SystemExit("No known tags. Check IDs/family/size/world JSON.")
        print("Used tag IDs:",ids); print("Camera position (m):",t_wc.ravel()); print("Quat (x,y,z,w):",R_to_quat(R_wc))
        draw_dets(img,dets,args.scale)
        if args.draw_axes:
            img=draw_axes(img,K,axes_dist,R_wc,t_wc,0.05)
        cv.imshow("pose",img); cv.waitKey(0); cv.destroyAllWindows(); return

    cap=open_cam(args.cam,args.backend)
    if not cap.isOpened(): raise SystemExit("Camera failed to open. Try --backend msmf or another --cam.")
    axes_dist = zero_dist
    map1 = map2 = None
    writer=None
    snap_dir=None
    if args.snapshot_dir:
        snap_dir = Path(args.snapshot_dir)
        snap_dir.mkdir(parents=True,exist_ok=True)
    last_frame_ud=None
    if args.csv:
        p=Path(args.csv); p.parent.mkdir(parents=True,exist_ok=True)
        writer=csv.writer(open(p,"w",newline="")); writer.writerow(["timestamp","x","y","z","qx","qy","qz","qw","num_tags"])

    print("Press q to quit. space=detection on/off")
    enable_detect=True
    last_R,last_t,last_ids=None,None,[]
    pose_alpha=float(np.clip(args.pose_ema,0.0,1.0))
    f=0
    while True:
        # flush buffer a bit (reduces single-frame “stuck”)
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break

        frame_ud=frame
        if undistort_frames:
            if map1 is None:
                h,w=frame.shape[:2]
                map1,map2=cv.initUndistortRectifyMap(K,dist,None,K,(w,h),cv.CV_32FC1)
            frame_ud=cv.remap(frame,map1,map2,interpolation=cv.INTER_LINEAR)
        vis=frame_ud.copy()
        gray=cv.cvtColor(frame_ud,cv.COLOR_BGR2GRAY)
        if args.snapshot_dir:
            last_frame_ud = frame_ud.copy()
        cv.putText(vis,f'frame mean:{gray.mean():.1f}',(10,55),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        run_this_frame = enable_detect and (f % max(1, args.detect_every) == 0)
        if run_this_frame:
            R_wc,t_wc,ids,dets=detect_and_pose(gray,cam_K,zero_dist if undistort_frames else dist,tag_size,world,detector,scale=args.scale,target_tag=args.only_tag)
            if R_wc is not None:
                last_R,last_t = blend_pose(last_R,last_t,R_wc,t_wc,pose_alpha)
                last_ids = ids
            draw_dets(vis,dets,args.scale)

        if last_t is not None:
            if args.draw_axes:
                vis=draw_axes(vis,K,axes_dist,last_R,last_t,0.05)
            qx,qy,qz,qw=R_to_quat(last_R)
            cv.putText(vis,f"tags:{len(last_ids)}  x:{last_t[0,0]:.3f} y:{last_t[1,0]:.3f} z:{last_t[2,0]:.3f} m",
                       (10,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            if writer and run_this_frame:
                writer.writerow([time.time(),last_t[0,0],last_t[1,0],last_t[2,0],qx,qy,qz,qw,len(last_ids)])

        cv.imshow("AprilTag localization", vis)
        key=cv.waitKey(1)&0xFF
        if key==ord('q'): break
        if key==ord(' '): enable_detect = not enable_detect
        if key==ord('p') and snap_dir is not None and last_t is not None and last_frame_ud is not None:
            ts=datetime.now().strftime('%Y%m%d_%H%M%S')
            img_path=snap_dir / f'snapshot_{ts}.png'
            pose_path=snap_dir / f'snapshot_{ts}.json'
            cv.imwrite(str(img_path), last_frame_ud)
            pose_info = {
                'timestamp': time.time(),
                'translation_m': [float(last_t[0,0]), float(last_t[1,0]), float(last_t[2,0])],
                'quaternion_xyzw': list(R_to_quat(last_R)),
                'used_tags': last_ids,
                'params_file': args.params,
                'world_file': args.world,
                'frame_undistorted': bool(undistort_frames),
                'undistort_alpha': float(args.undistort_alpha),
                'cam_matrix': cam_K.tolist()
            }
            pose_path.write_text(json.dumps(pose_info, indent=2))
            print(f"[snapshot] saved {img_path} and {pose_path}")
        f+=1

    cap.release(); cv.destroyAllWindows()

if __name__=="__main__": main()

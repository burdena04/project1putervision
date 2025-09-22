# partA_min.py — meets Part A, minimal features, external webcam (index 1)
# keys: c = mosaic toggle, s = save PNG/JPEGs, p = pixel probe, q/ESC = quit

import cv2 as cv
import numpy as np
import os
from datetime import datetime

OUT_DIR = "partA_outputs"
CAM_INDEX = 1  # external webcam; change to 0 if you want the laptop cam

def ensure_dir(path): os.makedirs(path, exist_ok=True)

def save_png(bgr):
    """Save a single PNG (lossless). Returns list with the saved path."""
    ensure_dir(OUT_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"frame_{ts}.png")
    cv.imwrite(path, bgr)
    print("Saved PNG:\n ", path)
    return [path]


def save_jpeg(bgr, quality=95):
    """Save a single JPEG at the given quality (default 95). Returns list with the saved path."""
    ensure_dir(OUT_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"frame_{ts}.jpg")
    cv.imwrite(path, bgr, [int(cv.IMWRITE_JPEG_QUALITY), quality])
    print(f"Saved JPEG (q={quality}):\n ", path)
    return [path]

def color_mosaic(bgr):
    rgb  = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    hsv  = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    v3   = cv.cvtColor(hsv[:,:,2], cv.COLOR_GRAY2BGR)
    g3   = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    top = np.hstack([bgr, rgb]); bot = np.hstack([g3, v3])
    m = np.vstack([top, bot])
    h, w = bgr.shape[:2]
    cv.putText(m, "BGR", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv.putText(m, "RGB", (w+10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv.putText(m, "GRAY", (10, h+28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv.putText(m, "HSV-V", (w+10, h+28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return m

class PixelProbe:
    def __init__(self): self.info = None
    def update(self, bgr):
        self.bgr = bgr
        self.hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        self.rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        self.gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            b,g,r = self.bgr[y,x].tolist()
            h,s,v = self.hsv[y,x].tolist()
            rr,gg,bb = self.rgb[y,x].tolist()
            gr = int(self.gray[y,x])
            self.info = (x,y,(b,g,r),(rr,gg,bb),(h,s,v),gr)
            print(f"({x},{y}) BGR={b,g,r}  RGB={rr,gg,bb}  HSV={h,s,v}  GRAY={gr}")
    def annotate(self, img):
        if not self.info: return img
        x,y,_,_,hsv,gr = self.info
        out = img.copy()
        cv.circle(out,(x,y),6,(0,255,255),2)
        cv.rectangle(out,(10,10),(910,42),(0,0,0),-1)
        cv.putText(out, f"({x},{y}) HSV={hsv} GRAY={gr}", (16,34),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv.LINE_AA)
        return out

def open_cam(idx):
    for backend in (cv.CAP_MSMF, cv.CAP_DSHOW, cv.CAP_ANY):
        cap = cv.VideoCapture(idx, backend)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            return cap
        cap.release()
    raise RuntimeError(f"Cannot open camera index {idx}")

def main():
    cap = open_cam(CAM_INDEX)
    win = "Part A — Live (c mosaic, s=save PNG, j=save JPG, p probe, q quit)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    probe = PixelProbe()
    cv.setMouseCallback(win, probe.on_mouse)

    show_mosaic = False
    probe_mode = False

    print("keys: c=mosaic  s=save PNG  j=save JPG  p=probe  q/ESC=quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None: break

        display = color_mosaic(frame) if show_mosaic else frame
        if probe_mode:
            probe.update(frame)
            display = probe.annotate(display)

        cv.imshow(win, display)
        k = cv.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == ord('c'):
            show_mosaic = not show_mosaic
        elif k == ord('s'):
            save_png(frame)
        elif k == ord('j'):
            save_jpeg(frame)
        elif k == ord('p'):
            probe_mode = not probe_mode

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

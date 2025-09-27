# capture_calib_images.py
import cv2 as cv
import os, time, argparse, textwrap

def main():
    ap = argparse.ArgumentParser(
        description="Capture chessboard images for calibration",
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="calib_images", help="output folder")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column (rows)")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row (cols)")
    ap.add_argument("--interval", type=float, default=0.5, help="min seconds between saves")
    ap.add_argument("--width", type=int, default=1920, help="capture width in pixels")
    ap.add_argument("--height", type=int, default=1080, help="capture height in pixels")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv.VideoCapture(args.cam, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Camera failed to open. Try a different --cam index.")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    pattern_size = (args.cols, args.rows)
    last_save = 0.0
    count = 0
    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera streaming at {actual_width}x{actual_height}")
    print(textwrap.dedent(f"""
        Controls:
          s : save current frame (only if pattern is confidently found)
          g : force save (even if corners not found) -- use sparingly
          q : quit
        Hint: vary angle, distance, position; fill different parts of the image.
    """))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, pattern_size,
                                                flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE)
        vis = frame.copy()
        if ret:
            # refine for display
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            cv.drawChessboardCorners(vis, pattern_size, corners, ret)

        cv.putText(vis, f"{args.rows}x{args.cols} inner corners", (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.imshow("capture_calib_images", vis)
        k = cv.waitKey(1) & 0xFF
        now = time.time()

        if k == ord('q'):
            break
        elif k == ord('s') and ret and (now - last_save) >= args.interval:
            path = os.path.join(args.out, f"cb_{count:03d}.png")
            cv.imwrite(path, frame)
            print(f"[OK] saved {path}")
            count += 1
            last_save = now
        elif k == ord('g') and (now - last_save) >= args.interval:
            path = os.path.join(args.out, f"forced_{count:03d}.png")
            cv.imwrite(path, frame)
            print(f"[warn] forced save {path} (no pattern check)")
            count += 1
            last_save = now

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

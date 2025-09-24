# cam_sanity.py
import cv2 as cv, time
cap = cv.VideoCapture(0, cv.CAP_MSMF)  # try MSMF first on Windows
if not cap.isOpened():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    print("fallback to DSHOW")
if not cap.isOpened():
    cap = cv.VideoCapture(0)
    print("fallback to default")
t0, n = time.time(), 0
while True:
    ok, frame = cap.read()
    if not ok: print("read failed"); break
    n += 1
    if n % 30 == 0:
        fps = n / (time.time() - t0)
        print(f"fps ~ {fps:.1f}")
    cv.imshow("cam", frame)
    if (cv.waitKey(1) & 0xFF) == ord('q'):
        break
cap.release(); cv.destroyAllWindows()

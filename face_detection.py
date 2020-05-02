import argparse
import numpy as np
import cv2
import dlib
import dcv2lib

# ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face-predictor', required=True, help='Path to facial landmark predictor')
ap.add_argument('-i', '--image', required=False, help='Path to an image (replaces camera)')
ap.add_argument('-v', '--video', required=False, help='Path to a video or camera index', default='0')
ap.add_argument('-b', '--blackmode', required=False, help='Overwrites the blackmode')
args = ap.parse_args()

# Face detection/prediction
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(args.face_predictor)

if args.blackmode:
    if args.blackmode == '0':
        blackmode = False
    else:
        blackmode = True
else:
    blackmode = False

enable_rect = True
enable_lines = True
enable_points = True
enable_ear = True

def drawlines(img, shape, name=''):
    sn = 0
    for (sx, sy) in shape:
        if sn < len(shape)-1:
            nx, ny = shape[sn+1]
            cv2.line(img, (sx, sy), (nx, ny), (0, 0, 255), 2)
            sn += 1
        else:
            nx, ny = shape[0]
            cv2.line(img, (sx, sy), (nx, ny), (0, 0, 255), 2)

def detect_faces(img_orig, img, img_gray):
    global last_frame
    rects = face_detector(img_gray, 1)
    for (i, rect) in enumerate(rects):
        shape = dcv2lib.shape_to_np(face_predictor(img_gray, rect), 68)
        (x, y, w, h) = dcv2lib.rect_to_bb(rect)
        if enable_rect:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, 'Face #' + str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if enable_lines:
            for area in dcv2lib.FL68AREAS:
                start, end = dcv2lib.FL68AREAS[area]
                drawlines(img, shape[start:end], area)
        if enable_points:
            for (sx, sy) in shape:
                cv2.circle(img, (sx, sy), 1, (255, 0, 255), -1)
        
        # EAR detection
        if enable_ear:
            (lestart, leend) = dcv2lib.FL68AREAS['left_eye']
            (restart, reend) = dcv2lib.FL68AREAS['right_eye']
            left_eye = shape[lestart:leend]
            right_eye = shape[restart:reend]
            left_ear = dcv2lib.get_ear(left_eye)
            right_ear = dcv2lib.get_ear(right_eye)
            ear = (left_ear + right_ear) / 2
            cv2.putText(img, 'EAR: ' + str(ear), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if len(rects) > 0:
        if blackmode:
            last_frame = img
        return img, True
    else:
        if blackmode:
            return last_frame, False
        return img, False

def detect(img_orig):
    global last_frame
    h, w, c = img_orig.shape
    if blackmode:
        img = np.zeros((h, w, c), np.uint8)
    else:
        img = img_orig.copy()
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    try:
        last_frame
    except NameError:
        last_frame = np.zeros((h, w, c), np.uint8)
    img, success = detect_faces(img_orig, img, img_gray)
    cv2.imshow('Facial Landmark Detection #' + args.video, img)
    return success

if args.image:
    detect(cv2.imread(args.image))
    cv2.waitKey(0)
    print('Closed by user')
else:
    try:
        cam = cv2.VideoCapture(int(args.video))
        flip = True
    except ValueError:
        cam = cv2.VideoCapture(args.video)
        flip = False
    if not cam.isOpened():
        print('Could not open video capture')
    while cam.isOpened():
        key = cv2.waitKey(1)
        if key == 27:
            print('Closed by user')
            break
        elif key == 122 and not args.blackmode:
            blackmode = not blackmode
        elif key == 49:
            enable_rect = not enable_rect
        elif key == 50:
            enable_lines = not enable_lines
        elif key == 51:
            enable_points = not enable_points
        elif key == 52:
            enable_ear = not enable_ear
        success, img_orig = cam.read()
        if not success:
            print('Could not get image from video capture')
            return
        if flip:
            img_orig = cv2.flip(img_orig, 1)
        detect(img_orig)
cv2.destroyAllWindows()

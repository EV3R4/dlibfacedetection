import argparse
import json
import os

import cv2
import numpy as np

import dcv2lib
import dlib

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=False, help='Path to the config file.', default='config.json')
ap.add_argument('-f', '--face-predictor', required=True, help='Path to facial landmark predictor.')
ap.add_argument('-i', '--image', required=False, help='Path to an image (replaces -v/--video).')
ap.add_argument('-v', '--video', required=False, help='Path to a video or camera index.', default='0')
ap.add_argument('-b', '--blackmode', required=False, help='Forcefully activates the blackmode.', action="store_true")
ap.add_argument('-V', '--no-video-window', required=False, help='Deactivates the video window.', action="store_true")
ap.add_argument('-I', '--no-input-window', required=False, help='Deactivates the input window.', action="store_true")
args = ap.parse_args()

DEFAULT_CONFIG ='''{
    "enable_last_frame": false,
    "rect_color": [0, 255, 0],
    "line_color": [0, 0, 255],
    "point_color": [255, 0, 255],
    "text_color": [0, 255, 0],
    "ear_color": [0, 255, 0],
    "f2r_color": [255, 255, 255],
    "double_f2r_mouth_height": false
}'''
CONFIG_FORMAT = {
    'enable_last_frame': bool,
    'rect_color': list,
    'line_color': list,
    'point_color': list,
    'text_color': list,
    'ear_color': list,
    'f2r_color': list,
    'double_f2r_mouth_height': bool
}
def load_config():
    global config
    if not os.path.isfile(args.config):
        with open(args.config, 'w') as f:
            f.write(DEFAULT_CONFIG)
    try:
        with open(args.config, 'r') as f:
            config = json.loads(f.read())
    except json.JSONDecodeError:
        print('Error: Config is malformed')
        exit(1)
    
    # Validation
    contains = []
    for key, value in config.items():
        try:
            if type(value) is not CONFIG_FORMAT[key]:
                print('Error: Config value "' + key + '" needs to be of type ' + CONFIG_FORMAT[key].__name__)
                exit(1)
            else:
                contains.append(key)
        except KeyError:
            print('Warning: Config contains unknown value "' + key + '"')
    for key in CONFIG_FORMAT:
        if not key in config:
            print('Error: Config is missing "' + key + '"')
            exit(1)

load_config()

# Face detection/prediction
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(args.face_predictor)

blackmode = args.blackmode

enable_rect = True
enable_lines = True
enable_points = True
enable_text = True
enable_ear = True
enable_f2r = False

def detect_faces(img_gray):
    rects = face_detector(img_gray, 1)
    shapes = []
    for rect in rects:
        shape = dcv2lib.shape_to_np(face_predictor(img_gray, rect), 68)
        shapes.append(shape)
    return rects, shapes

def draw_rect(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), config['rect_color'], 2)

def draw_text(img, text, pos, color=config['text_color']):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_ear(img, shape, x, y, h):
    (lestart, leend) = dcv2lib.FL68AREAS['left_eye']
    (restart, reend) = dcv2lib.FL68AREAS['right_eye']
    left_eye = shape[lestart:leend]
    right_eye = shape[restart:reend]
    left_ear = dcv2lib.get_ear(left_eye)
    right_ear = dcv2lib.get_ear(right_eye)
    ear = (left_ear + right_ear) / 2
    draw_text(img, 'EAR: ' + str(ear), (x, y + h + 20), config['ear_color'])

def draw_lines(img, shape):
    for area in dcv2lib.FL68AREAS:
        start, end = dcv2lib.FL68AREAS[area]
        shape_area = shape[start:end]

        sn = 0
        for (sx, sy) in shape_area:
            if sn < len(shape_area)-1:
                nx, ny = shape_area[sn+1]
                cv2.line(img, (sx, sy), (nx, ny), config['line_color'], 2)
                sn += 1
            else:
                nx, ny = shape_area[0]
                cv2.line(img, (sx, sy), (nx, ny), config['line_color'], 2)

def draw_points(img, shape):
    for (sx, sy) in shape:
        cv2.circle(img, (sx, sy), 1, config['point_color'], -1)

def draw_f2r(img, shape):
    if config['double_f2r_mouth_height']:
        mouth_height = (shape[64][0], shape[66][1] + (shape[66][1] - shape[62][1]) * 2)
    else:
        mouth_height = (shape[64][0], shape[66][1])
    cv2.rectangle(img, (shape[60][0], shape[62][1]), mouth_height, config['f2r_color'], 2)
    cv2.rectangle(img, (shape[36][0], int((shape[37][1] + shape[38][1]) / 2)), (shape[39][0], int((shape[41][1] + shape[40][1]) / 2)), config['f2r_color'], 2)
    cv2.rectangle(img, (shape[42][0], int((shape[43][1] + shape[44][1]) / 2)), (shape[45][0], int((shape[47][1] + shape[46][1]) / 2)), config['f2r_color'], 2)

def draw_faces(img, rects, shapes):
    for i, rect in enumerate(rects):
        (x, y, w, h) = dcv2lib.rect_to_bb(rect)
        shape = shapes[i]

        if enable_rect:
            draw_rect(img, x, y, w, h)
        if enable_lines:
            draw_lines(img, shape)
        if enable_points:
            draw_points(img, shape)
        if enable_text:
            draw_text(img, 'Face #' + str(i), (x, y - 10))
        if enable_ear:
            draw_ear(img, shape, x, y, h)
        if enable_f2r:
            draw_f2r(img, shape)

def detect_and_show(img, img_gray):
    h, w, c = img.shape
    
    rects, shapes = detect_faces(img_gray)

    if blackmode:
        img = np.zeros((h, w, c), np.uint8)

    draw_faces(img, rects, shapes)

    if len(rects) > 0 or not config['enable_last_frame']:
        cv2.imshow('Facial Landmark Detection #' + args.video, img)

def handle_input():
    global blackmode, enable_rect, enable_lines, enable_points, enable_text, enable_ear, enable_f2r

    key = cv2.waitKey(1)
    if key == 27:
        print('Closed by user')
        return False
    elif key == 114:
        load_config()
    elif key == 122 and not args.blackmode:
        blackmode = not blackmode
    elif key == 49:
        enable_rect = not enable_rect
    elif key == 50:
        enable_lines = not enable_lines
    elif key == 51:
        enable_points = not enable_points
    elif key == 52:
        enable_text = not enable_text
    elif key == 53:
        enable_ear = not enable_ear
    elif key == 54:
        enable_f2r = not enable_f2r

    return True

if args.image:
    img = cv2.imread(args.image)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if not args.no_video_window: cv2.imshow('Video #' + args.video, img)

    if not args.no_input_window: cv2.imshow('Input #' + args.video, img_gray)

    while handle_input():
        detect_and_show(img.copy(), img_gray)
else:
    try:
        vid = cv2.VideoCapture(int(args.video))
        flip = True
    except ValueError:
        vid = cv2.VideoCapture(args.video)
        flip = False
    if not vid.isOpened():
        print('Could not open video capture')
    while vid.isOpened() and handle_input():
        success, img = vid.read()
        if not success:
            print('Could not get image from video capture')
            break

        if flip:
            img = cv2.flip(img, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if not args.no_video_window: cv2.imshow('Video #' + args.video, img)
    
        if not args.no_input_window: cv2.imshow('Input #' + args.video, img_gray)

        detect_and_show(img, img_gray)
    vid.release()
cv2.destroyAllWindows()

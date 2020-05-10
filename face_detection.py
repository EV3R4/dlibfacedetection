from argparse import ArgumentParser
import numpy as np
import cv2
import dlib
import dcv2lib
from json import loads
from json.decoder import JSONDecodeError
from os.path import isfile
from shutil import copyfile
from sys import exit

# ArgumentParser
ap = ArgumentParser()
ap.add_argument('-c', '--config', required=False, help='Path to the config file', default='config.json')
ap.add_argument('-f', '--face-predictor', required=True, help='Path to facial landmark predictor')
ap.add_argument('-i', '--image', required=False, help='Path to an image (replaces -v/--video)')
ap.add_argument('-v', '--video', required=False, help='Path to a video or camera index', default='0')
ap.add_argument('-b', '--blackmode', required=False, help='Overwrites the blackmode')
args = ap.parse_args()

DEFAULT_CONFIG ='''{
    "enable_last_frame": true,
    "rect_color": [0, 255, 0],
    "text_color": [0, 255, 0],
    "ear_color": [0, 255, 0],
    "line_color": [0, 0, 255],
    "point_color": [255, 0, 255],
    "f2r_color": [255, 255, 255],
    "double_f2r_mouth_height": false
}'''
CONFIG_FORMAT = {
    'enable_last_frame': bool,
    'rect_color': list,
    'text_color': list,
    'ear_color': list,
    'line_color': list,
    'point_color': list,
    'f2r_color': list,
    'double_f2r_mouth_height': bool
}
def load_config():
    global config
    if not isfile(args.config):
        with open(args.config, 'w') as f:
            f.write(DEFAULT_CONFIG)
    try:
        with open(args.config, 'r') as f:
            config = loads(f.read())
    except JSONDecodeError:
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
FACE_DETECTOR = dlib.get_frontal_face_detector()
FACE_PREDICTOR = dlib.shape_predictor(args.face_predictor)

if args.blackmode:
    if args.blackmode == '0':
        blackmode = False
    else:
        blackmode = True
else:
    blackmode = False

enable_rect = True
enable_text = True
enable_ear = True
enable_lines = True
enable_points = True
enable_f2r = False

def drawlines(img, shape, name=''):
    sn = 0
    for (sx, sy) in shape:
        if sn < len(shape)-1:
            nx, ny = shape[sn+1]
            cv2.line(img, (sx, sy), (nx, ny), config['line_color'], 2)
            sn += 1
        else:
            nx, ny = shape[0]
            cv2.line(img, (sx, sy), (nx, ny), config['line_color'], 2)

def detect_faces(img_orig, img, img_gray):
    global last_frame
    rects = FACE_DETECTOR(img_gray, 1)
    for (i, rect) in enumerate(rects):
        shape = dcv2lib.shape_to_np(FACE_PREDICTOR(img_gray, rect), 68)
        (x, y, w, h) = dcv2lib.rect_to_bb(rect)
        if enable_rect:
            cv2.rectangle(img, (x, y), (x+w, y+h), config['rect_color'], 2)
        if enable_text:
            cv2.putText(img, 'Face #' + str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['text_color'], 2)
        if enable_ear:
            (lestart, leend) = dcv2lib.FL68AREAS['left_eye']
            (restart, reend) = dcv2lib.FL68AREAS['right_eye']
            left_eye = shape[lestart:leend]
            right_eye = shape[restart:reend]
            left_ear = dcv2lib.get_ear(left_eye)
            right_ear = dcv2lib.get_ear(right_eye)
            ear = (left_ear + right_ear) / 2
            cv2.putText(img, 'EAR: ' + str(ear), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['ear_color'], 2)
        if enable_lines:
            for area in dcv2lib.FL68AREAS:
                start, end = dcv2lib.FL68AREAS[area]
                drawlines(img, shape[start:end], area)
        if enable_points:
            for (sx, sy) in shape:
                cv2.circle(img, (sx, sy), 1, config['point_color'], -1)
        if enable_f2r:
            if config['double_f2r_mouth_height']:
                mouth_height = (shape[64][0], shape[66][1] + (shape[66][1] - shape[62][1]) * 2)
            else:
                mouth_height = (shape[64][0], shape[66][1])
            cv2.rectangle(img, (shape[60][0], shape[62][1]), mouth_height, config['f2r_color'], 2)
            cv2.rectangle(img, (shape[36][0], int((shape[37][1] + shape[38][1]) / 2)), (shape[39][0], int((shape[41][1] + shape[40][1]) / 2)), config['f2r_color'], 2)
            cv2.rectangle(img, (shape[42][0], int((shape[43][1] + shape[44][1]) / 2)), (shape[45][0], int((shape[47][1] + shape[46][1]) / 2)), config['f2r_color'], 2)
    
    if len(rects) > 0:
        if blackmode and config['enable_last_frame']:
            last_frame = img
        return img, True
    else:
        if blackmode and config['enable_last_frame']:
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
        vid = cv2.VideoCapture(int(args.video))
        flip = True
    except ValueError:
        vid = cv2.VideoCapture(args.video)
        flip = False
    if not vid.isOpened():
        print('Could not open video capture')
    while vid.isOpened():
        key = cv2.waitKey(1)
        if key == 27:
            print('Closed by user')
            break
        elif key == 114:
            load_config()
        elif key == 122 and not args.blackmode:
            blackmode = not blackmode
        elif key == 49:
            enable_rect = not enable_rect
        elif key == 50:
            enable_text = not enable_text
        elif key == 51:
            enable_ear = not enable_ear
        elif key == 52:
            enable_lines = not enable_lines
        elif key == 53:
            enable_points = not enable_points
        elif key == 54:
            enable_f2r = not enable_f2r
        success, img_orig = vid.read()
        if not success:
            print('Could not get image from video capture')
            break
        if flip:
            img_orig = cv2.flip(img_orig, 1)
        detect(img_orig)
    vid.release()
cv2.destroyAllWindows()

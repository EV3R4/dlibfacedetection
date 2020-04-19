'''
    dcv2lib

    MIT License

    Copyright (c) 2020 EV3R4

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

import numpy as np
import dlib
from scipy.spatial import distance as dist

FL68AREAS = {
    'jaw': (0, 17),
    'mouth': (48, 68),
    'nose': (27, 36),
    'left_eye': (36, 42),
    'right_eye': (42, 48),
    'left_eyebrow': (17, 22),
    'right_eyebrow': (22, 27)
}

def bb_to_rect(bb):
    return dlib.rectangle(bb.x, bb.y, bb.w, bb.h)

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, size, dtype='int'):
    coords = np.zeros((size, 2), dtype=dtype)
    for i in range(0, size):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def bb_contains(bb, point):
    if point[0] < bb[0]:
        return False
    elif point[1] < bb[1]:
        return False
    elif point[0] < bb[2]:
        return False
    elif point[1] < bb[3]:
        return False
    return True

def rect_contains(rect, point):
    return bb_contains(rect, point)

def get_ear(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    return (a+b) / (2*c)

# OpenCV Face Detection
![GitHub](https://img.shields.io/github/license/EV3R4/OpenCVFaceDetection)
![GitHub repo size](https://img.shields.io/github/repo-size/EV3R4/OpenCVFaceDetection)

Just some face detection code.

## Installation
* Install [Python](https://www.python.org/)
* Clone the repository with `git clone https://github.com/EV3R4/OpenCVFaceDetection.git`
* Install the requirements with `pip install -r requirements.txt`
* Download the [pretrained model](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)

## Executing the face detection
Run `python face_detection.py` with the needed arguments

The value of `-f` needs to be set to the model "shape_predictor_68_face_landmarks.dat"

If you set the value `-b` to 0 or 1 you will overwrite the "blackmode" value

In the "blackmode" the background will be black

## Keys
* ESCAPE - Closes the window
* z - Enables/Disables "blackmode"
* 1 - Enables/Disables rects around the faces
* 2 - Enables/Disables lines
* 3 - Enables/Disables points
* 4 - Enables/Disables the EAR display

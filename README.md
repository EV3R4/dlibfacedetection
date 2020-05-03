# dlib face detection
## Just some face detection code.
![GitHub](https://img.shields.io/github/license/EV3R4/dlibfacedetection)
![GitHub repo size](https://img.shields.io/github/repo-size/EV3R4/dlibfacedetection)

![preview](preview_rescaled.png)

## Installation
* Install [Python](https://www.python.org/)
* Clone the repository with `git clone https://github.com/EV3R4/dlibfacedetection.git` or download the [zip](https://github.com/EV3R4/dlibfacedetection/archive/master.zip)
* Install the requirements with `pip install -r requirements.txt`
* Download the [pretrained model](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)

## Executing the face detection
Run `python face_detection.py` with the needed arguments

The value of `-f` needs to be set to the model "shape_predictor_68_face_landmarks.dat"

If you set the value `-b` to 0 or 1 you will overwrite the "blackmode" value

In the "blackmode" the background will be black which is useful for streaming

## Keys
* ESCAPE - Closes the window
* z - Enables/Disables "blackmode"
* 1 - Enables/Disables rects around the faces
* 2 - Enables/Disables lines
* 3 - Enables/Disables points
* 4 - Enables/Disables the EAR display
* 5 - Enables/Disables Face to Rectangles (F2R)

## Config
### Values
* rect_color: The color of rects around faces
* line_color: The color of lines
* point_color: The color of points
* f2r_color: The color of Face to Rectangles (F2R)

## Notes
If no faces were found the program uses the last frame were it found faces

The program might not work correctly if you cover parts of your face or wear glasses

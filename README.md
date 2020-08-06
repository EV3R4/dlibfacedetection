# dlib face detection
## Face detection display for dlib
![GitHub](https://img.shields.io/github/license/EV3R4/dlibfacedetection)
![GitHub repo size](https://img.shields.io/github/repo-size/EV3R4/dlibfacedetection)

![preview](preview_rescaled.png)

## Installation
* Install [Python](https://www.python.org/).
* Clone the repository with `git clone https://github.com/EV3R4/dlibfacedetection.git` or download the [zip](https://github.com/EV3R4/dlibfacedetection/archive/master.zip).
* Install the requirements with `pip install -r requirements.txt`.
* Download the [pretrained model](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2).

## Executing the face detection
Run `python face_detection.py` with the needed arguments.

The value of `-f/--face-predictor` needs to be set to the model "shape_predictor_68_face_landmarks.dat".

If the `-b/--blackmode` argument is passed blackmode will be forcefully activated.

In blackmode the background (image from the camera) will be black which is useful for streaming or other cases where you want to hide your background or face.

## Keys
* ESCAPE - Closes the window
* r - Reloads the config
* z - Enables/disables blackmode
* 1 - Enables/disables rectangles around faces
* 2 - Enables/disables lines
* 3 - Enables/disables points
* 4 - Enables/disables face indexes over rectangles
* 5 - Enables/disables Eye Aspect Ratio (EAR) under rectangles
* 6 - Enables/disables Face to Rectangles (F2R)

## Config
### Values
* enable_last_frame: If set to `true`, the program uses the last frame in which faces were found if no faces are being detected.
* rect_color: The color of rectangles around the faces.
* text_color: The color of face numbers.
* ear_color: The color of the Eye Aspect Ratio (EAR) text.
* line_color: The color of lines.
* point_color: The color of points.
* f2r_color: The color of Face to Rectangles (F2R).
* double_f2r_mouth_height: If set to `true` doubles the mouth height.

## Face to Rectangles (F2R)
![preview](f2r_preview_rescaled.png)

## Notes
The program might not work correctly if you cover parts of your face, wear glasses or have hair in front of your face.

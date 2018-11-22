# OpenCV Upload Validation API

## Introductions
This is image processing for upload validation again some requirements focused on content on uploaded image. Using `Python` and `OpenCV` as opensource language and libraries.
Each object (face, face & upperbody, left-right eye, nose, mouth) detections are now using `Haarcascade`. `Haarcascade` files stored in `haarcascades` directory. 
Another further and advanced solutions for better object detections can use `Dlib`, but need more time to explore that library.

## Requirements & Solutions
* **Good quality photo, less than 6 months old ?**

	Solutions: 
    > Get metadata from the image file by reading the `EXIF 'tag'`.

* **Clear, focused image with no marks or �red�-eye**

	Solutions: 
	> **_Clear & focused_**: extract main area of the photo (face), scale it then convert to gray scale color. Finally detect blury area by openCV `Laplacian` function.

	> **_Red Eye_**: extract two areas (left eye & right eye), keep the color, blur it so the area are smoothed, and then convert from `Blue, Green, Red (BGR)` color to `HSV (Hue, Saturation, Value)`. Each area then processed by detecting contour per pixel by range of "lower-upper" of (red eye color BRG color).

* **Head or head and shoulders (upperbody)**

	Solutions: 
    
    > Detect face or face and upperbody using haarcascades (face & upperbody) by some iamge scale and ratio definition will match again upperbody ratio.

* **Face looking at the camera (face detection: left eye, right eye, nose, mouth)**

	Solutions:
	> Detect and extract face area in the image then split it into four parts by `X-Y coordinate` (top-middle-left, top-middle-right, middle-middle, bottom-middle).
	> Each area must contains object like left-right eye, nose, and mouth.
	
		- Top-middle-left: left-eye
		- Top-middle-right: right-eye
		- Middle-middle: nose
		- Bottom-middle: mouth

	> Must be found at least one left-right eye and one nose also one mouth.

* **Plain background preferably white or light grey ?**

	Solutions: 
    
    > Populate and count color each pixel the the most "common" color in the image, the most color of RGB value most larger than `rgb(127,127,127)`. RGB of gray color is `rgb (128,128,128)`.

Installations:
## Minimum System Requirements:
	Ubuntu 16.04
	Python 2.7
	OpenCV 2.4.9.1
	Django REST Framework 3.9.0

## Recommended System Requirements:
	Ubuntu 18.04
	Python 3+
	OpenCV 3+

## Installing Python and OpenCV

```
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python-opencv
$ sudo apt install python-pip
```

## Check OpenCV installation

```
$ python3
>>> import cv2
>>> cv2.__version__
'3.2.0'
```

(Press Ctrl+D to Exit)

## Installing Python ExifRead Libray
```
$ pip install exifread
```

## Installing Python Pillow Libray
```
$ pip install pillow
```

## Installing Djangorestframework
```
$ pip install numpy django requests
$ pip install djangorestframework
$ pip install markdown
$ pip install django-filter
```

## Upload and Unzip opencv_upload_vaidation_api.zip
```
$ cd opencv-upload-validation-api
```

## Open and allow Port 8000
```
$ sudo ufw allow 8000
```

## Running Application
[WITH DJANGO REST FRAMEWORK]
```
$ python manage.py runserver 0.0.0.0:8000
```
```
Performing system checks...

System check identified no issues (0 silenced).
November 22, 2018 - 05:51:37
Django version 1.11.16, using settings 'opencv_upload_validation_api.settings'
Starting development server at http://0.0.0.0:8000/
Quit the server with CONTROL-C.
```

[WITHOUT DJANGO REST FRAMEWORK]
```
$ python validator.py samples/d.jpg
```


```
====================================
#      All Detetcion Results       #
====================================
Image was  taken at 2016:07:19 22:05:49 about 28 months ago!
Found 1 upperbody!
Found 1 face!
Found 3 eyes!
Found 1 nose!
Found 1 mouth!
Found 0 red eye!
Est. Background is rgb(255,0,255)!
====================================
#      Criteria Test Results       #
====================================
1.   It was taken less than six months? True
2.1. Is not blurry? True
2.2. Is red eye found? False
3.   Is head or upperbody found? True
4.   Is face looking at camera? True
5.   Is preferable background? False
====================================
#        Test Case Result          #
====================================
False
```

##### [LOCAL HOST]

CURL from remote file:
```
$ curl -X POST 'http://localhost:8000/validation/validate/' -d 'url=http://209.97.161.116/opencv_upload_validation_api/samples/d.jpg'
```
CURL from local file:
```
$ curl -X POST 'http://localhost:8000/validation/validate/' -F "image=@/mnt/c/xampp/htdocs/trisna-ashari/opencv_upload_validation_api/samples/d.jpg"
```

##### [LIVE HOST]
Example Live Server:
Endpoint: 
```
http://209.97.161.116:8000/validation/validate/
```

##### [EXAMPLE OUTPUT]:
Invalid or Bad request (no url or no file submitted)

	{"passes": false}
    
Valid request
	
    {"passes": false, 
		"desc": {
			"is_preferable_background": false, 
			"is_head_or_upperbody_found": true, 
			"is_not_blurry": true, 
			"is_face_looking_at_camera": true, 
			"is_less_than_six_months": false, 
			"is_date_creation_found": false, 
			"is_red_eye_found": false
		}
	}
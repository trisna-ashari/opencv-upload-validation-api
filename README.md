## Table of Contents
- [What is OpenCV Upload Validation API?](#what-is-opencv-upload-validation-api)
- [Requirements & Solutions](#requirements--solutions)
    - [1. Good quality photo, less than 6 months old](#1-good-quality-photo-less-than-6-months-old)
    - [2. Clear, focused image with no marks or red-eye](#2-clear-focused-image-with-no-marks-or-red-eye)
    - [3. Head or head and shoulders (upperbody)](#3-head-or-head-and-shoulders-upperbody)
    - [4. Face looking at the camera (face detection: left eye, right eye, nose, mouth)](#4-face-looking-at-the-camera-face-detection-left-eye-right-eye-nose-mouth)
    - [5. Plain background preferably white or light grey](#5-plain-background-preferably-white-or-light-grey)
- [Installation for Mac](#installation-for-mac)
- [Installation for Ubuntu](#installation-for-ubuntu)
- [Example Test Output](#example-test-output)
- [License](#license)
- [Copyright](#copyright)

## What is OpenCV Upload Validation API?
This is image processing for upload validation again some requirements focused on content on uploaded image. Using `Python` and `OpenCV` as opensource language and libraries.
Each object (face, face & upperbody, left-right eye, nose, mouth) detections are now using `Haarcascade`. `Haarcascade` files stored in `haarcascades` directory. 
Another further and advanced solutions for better object detections can use `Dlib`, but need more time to explore that library.

## Requirements & Solutions
#### 1. Good quality photo, less than 6 months old
* Photo taken from the camera contains `EXIF 'tag'`.
* Get metadata from the image file by reading the `EXIF 'tag'`.

#### 2. Clear, focused image with no marks or red-eye
* `Clear & focused`: extract main area of the photo (face), scale it then convert to gray scale color. Finally detect blury area by openCV `Laplacian` function.

* `Red Eye`: extract two areas (left eye & right eye), keep the color, blur it so the area are smoothed, and then convert from `Blue, Green, Red (BGR)` color to `HSV (Hue, Saturation, Value)`. Each area then processed by detecting contour per pixel by range of "lower-upper" of (red eye color BRG color).

#### 3. Head or head and shoulders (upperbody)
* Detect face or face and upperbody using haarcascades (face & upperbody) by some iamge scale and ratio definition will match again upperbody ratio.

#### 4. Face looking at the camera (face detection: left eye, right eye, nose, mouth)
* Detect and extract face area in the image then split it into four parts by `X-Y coordinate` (top-middle-left, top-middle-right, middle-middle, bottom-middle).
* Each area must contains object like left-right eye, nose, and mouth.

    - Top-middle-left: left-eye
    - Top-middle-right: right-eye
    - Middle-middle: nose
    - Bottom-middle: mouth
* Must be found at least one left-right eye and one nose also one mouth.

#### 5. Plain background preferably white or light grey
* Populate and count color each pixel the the most "common" color in the image
* The most color of RGB value most larger than `rgb(127,127,127)`. RGB of gray color is `rgb (128,128,128)`.


# Installation for Mac
System Requirements:
* XCode already installed

#### Step 1: Install Homebrew
Update Homebrew
```
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ brew update
```

Add Homebrew path in PATH
```
$ echo "# Homebrew" >> ~/.bash_profile
$ echo "export PATH=/usr/local/bin:$PATH" >> ~/.bash_profile
$ source ~/.bash_profile
```

#### Step 2: Install Python 2

Install python2
```
$ brew install python
$ brew link python
$ brew upgrade python
```

Check python path, it should output `/usr/local/bin/python2`
```
$ which python2
```

Check python version, it should output like `Python 2.7.16`
```
$ python2 --version
```

Add this command to `~/.bash_profile`, so you can call `python2` by type `python`
```
$ echo "export PATH=/usr/local/opt/python/libexec/bin:$PATH" >> ~/.bash_profile
```

#### Step 3: Install Python libraries in a Virtual Environment

Install virtual environment
```
$ pip install virtualenv virtualenvwrapper
$ echo "# Virtual Environment Wrapper"
$ echo "VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python2" >> ~/.bash_profile
$ echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bash_profile
$ source ~/.bash_profile
```

Create virtual environment
```
$ mkvirtualenv opencv-py2 -p python2
$ workon opencv-py2
```

Install python libraries within this virtual environment
```
$ pip install exifread numpy django requests djangorestframework markdown django-filter pillow scipy matplotlib scikit-image scikit-learn ipython pandas
```

Quit virtual environment
```
$ deactivate
```

#### Step 4: Install OpenCV
Install OpenCV
```
$ brew install opencv
```

Add OpenCV’s site-packages path to global site-packages
```
$ echo /usr/local/opt/opencv/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/opencv3.pth
```

Use this command to find out the path of OpenCV on your machine
```
$ find /usr/local/opt/opencv/lib/ -name cv2*.so
```

Make OpenCV3 Python symlink in our virtual environment
```
$ cd ~/.virtualenvs/facecourse-py2/lib/python2.7/site-packages/
$ ln -s /usr/local/opt/opencv/lib//python3.7/site-packages/cv2/python-3.7/cv2.cpython-37m-darwin.so cv2.so
```

Install opencv-python
```
$ pip install opencv-python==3.1.0.0
```

#### Step 5: Test OpenCV

Activate virtual environment
```
$ workon opencv-py2
$ python
$ >>> import cv2
$ >>> cv2.__version__
```

#### Step 6: Run Project
Test sample with validator
```
$ python2 validator.py samples/a.jpg
```

Run Django REST Framework
```
$ python2 manage.py runserver 0.0.0.0:8000
```

# Installation for Ubuntu
### Instant Install
You can run instant setup by running installation script on project root.
```
$ bash install.sh
```

### Manual Installations:
Minimum System Requirements:
* Ubuntu 16.04
* Python 2.7
* OpenCV 2.4.9.1
* Django REST Framework 3.9.0

Recommended System Requirements:
* Ubuntu 18.04
* Python 3+
* OpenCV 3+

#### Step 1: Install Python and OpenCV

Install OpenCV
```
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python-opencv
$ sudo apt install python-pip
```

Check OpenCV installation

```
$ python3
>>> import cv2
>>> cv2.__version__
'3.2.0'
```

(Press Ctrl+D to Exit)

#### Step 2: Install python libraries
Installing Python ExifRead Libray
```
$ pip install exifread
```

Installing Python Pillow Libray
```
$ pip install pillow
```

Installing Djangorestframework
```
$ pip install numpy django requests
$ pip install djangorestframework
$ pip install markdown
$ pip install django-filter
```

#### Step 3: Clone repository & setup project

Clone repository
```
$ git clone https://gitlab.com/trisnaashari/opencv-upload-validation-api.git
```

Enter project directory
```
$ cd opencv-upload-validation-api
```

Open and allow Port 8000
```
$ sudo ufw allow 8000
```

#### Step 4: Run project
Running Application with Django Rest Framework
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

Running Application without Django Rest Framework
```
$ python validator.py samples/d.jpg
```

# Example Test Output
Run test by validator
```
$ python validator.py samples/d.jpg

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

CURL from remote file:
```
$ curl -X POST 'http://localhost:8000/validation/validate/' -d 'url=http://www.example.com/image.jpg'
```
CURL from local file:
```
$ curl -X POST 'http://localhost:8000/validation/validate/' -F "image=@samples/d.jpg"
```

Upload an image to your live server at:
```
[POST] http://your_ip:8000/validation/validate/
```

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
	
## License

MIT License. See LICENSE for details.

## Copyright

Copyright (c) 2017-2019 Trisna Novi Ashari.
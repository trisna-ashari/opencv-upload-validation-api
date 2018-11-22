from __future__ import unicode_literals
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.exceptions import ParseError
from rest_framework.parsers import FileUploadParser
import urllib
import json
import cv2
import sys
import os
import numpy as np
import exifread
import datetime as dt

# import pytesseract
from collections import Counter
# from pytesseract import image_to_string
from PIL import Image

@csrf_exempt
def validate(request):
	# initialize the data dictionary to be returned by the request
	data = {"passes": False}
 
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = get_image(stream=request.FILES["image"])
			imagePath = Image.open(request.FILES["image"])
			
		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)
 
			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
 
			# load the image and convert
			image = get_image(url=url)
			imagePath = Image.open(urllib.urlopen(url))
 
		# convert the image to grayscale, load the face cascade detector,
		# and detect faces in the image
		image = image
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				
		# Haarcascades Path
		faceCascPath = "haarcascades/haarcascade_frontalface_alt.xml"
		leftEyeCascPath = "haarcascades/haarcascade_mcs_lefteye.xml"
		rightEyeCascPath = "haarcascades/haarcascade_mcs_righteye.xml"
		noseCascPath = "haarcascades/haarcascade_mcs_nose.xml"
		mouthCascPath = "haarcascades/haarcascade_mcs_mouth.xml"
		upperbodyCascPath = "haarcascades/haarcascade_mcs_upperbody.xml"

		# Create the haarcascade
		faceCascade = cv2.CascadeClassifier(faceCascPath)
		leftEyeCascade = cv2.CascadeClassifier(leftEyeCascPath)
		rightEyeCascade = cv2.CascadeClassifier(rightEyeCascPath)
		noseCascade = cv2.CascadeClassifier(noseCascPath)
		mouthCascade = cv2.CascadeClassifier(mouthCascPath)
		upperbodyCascade = cv2.CascadeClassifier(upperbodyCascPath)
		
		# Test Case
		time_taken_time=0
		time_taken_month=0
		is_date_creation_found=False
		is_less_than_six_months=False
		is_not_blurry=1
		is_text_found=0
		is_red_eye_found=0
		is_clear=0
		is_head_or_upperbody_found=0
		is_face_looking_at_camera=0
		is_preferable_background=0

		# Variables
		total_red_eyes=0
		total_text=0
		total_blur=0
		total_upperbodies=0
		total_faces=0
		total_eyes=0
		total_noses=0
		total_mouths=0
		
		# ##############################
		# Detect upperbody in the image
		# ##############################
		upperbody = upperbodyCascade.detectMultiScale(
			gray,
			scaleFactor = 1.1,
			minNeighbors = 5,
			minSize = (70, 70), # Min size for valid detection, changes according to video size or body size in the video.
			flags = cv2.CASCADE_SCALE_IMAGE
		)

		if len(upperbody) == 0:
			upperbody = upperbodyCascade.detectMultiScale(
				gray,
				scaleFactor = 1.05,
				minNeighbors = 5,
				minSize = (140, 80), # Min size for valid detection, changes according to video size or body size in the video.
				flags = cv2.CASCADE_SCALE_IMAGE
			)

		if len(upperbody) == 0:
			upperbody = upperbodyCascade.detectMultiScale(
				gray,
				scaleFactor = 1.1,
				minNeighbors = 5,
				minSize = (100, 80), # Min size for valid detection, changes according to video size or body size in the video.
				flags = cv2.CASCADE_SCALE_IMAGE
			)

		# Draw a rectangle around the upperbody
		for (x, y, w, h) in upperbody:
			roi_color = image[y:y+h, x:x+w]
			cv2.rectangle(image, (x, y), (x + w, y + h), (255,000,255),2)
			cv2.imwrite('results/result_upperbody.jpg', roi_color)
			total_upperbodies +=1

		# ##############################
		# Detect faces in the image
		# ##############################
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE
		)

		# Iterate over each face found
		for (x, y, w, h) in faces:
			total_faces +=1
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = image[y:y+h, x:x+w]
			face = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.imwrite('results/result_face.jpg', roi_color)

			# ############################
			# Blurry Face detections
			# ############################
			total_blur = cv2.Laplacian(gray, cv2.CV_64F).var()

			#Searching left eye in the 1/2 of the face from start left
			roi_gray = gray[y:y+h*2/3, x:x + w/2]
			roi_color = image[y:y+h*2/3, x:x + w/2]

			# ############################
			# Detect left eye in the image
			# ############################
			left_eye = leftEyeCascade.detectMultiScale(roi_gray)

			for (ex, ey, ew, eh) in left_eye:
				total_eyes +=1
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(100,100,100),2)
				cv2.imwrite('results/result_face_left_eye.jpg', roi_color)

				if red_eye_checker(roi_color, ex, ey, ew, eh, 'left'):
					total_red_eyes += 1

			#Searching right eye in the 1/2 of the face from start from middle to the right
			roi_gray = gray[y:y+h*2/3, x + w/2:x+w]
			roi_color = image[y:y+h*2/3, x + w/2:x+w]

			# #############################
			# Detect right eye in the image
			# #############################
			right_eye = rightEyeCascade.detectMultiScale(roi_gray)

			for (ex, ey, ew, eh) in right_eye:
				total_eyes +=1
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(100,100,100),2)
				cv2.imwrite('results/result_face_right_eye.jpg', roi_color)

				if red_eye_checker(roi_color, ex, ey, ew, eh, 'right'):
					total_red_eyes += 1

			#Searching noses
			roi_gray = gray[y + h/3:y + h*5/6, x + w/4:x + w*3/4]
			roi_color = image[y + h/3:y + h*5/6, x + w/4:x + w*3/4]

			# ############################
			# Detect nose in the image
			# ############################
			nose = noseCascade.detectMultiScale(roi_gray)

			for (nx, ny, nw, nh) in nose:
				total_noses += 1
				cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
				cv2.imwrite('results/result_face_nose.jpg', roi_color)

			#Searching mouth
			roi_gray = gray[y + h*2/3:y + h, x + w/4:x + w*3/4]
			roi_color = image[y + h*2/3:y + h, x + w/4:x + w*3/4]

			# ############################
			# Detect mouth in the image
			# ############################
			mouth = mouthCascade.detectMultiScale(roi_gray)

			for (mx, my, mw, mh) in mouth:
				total_mouths += 1
				cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,0),2)
				cv2.imwrite('results/result_face_mouth.jpg', roi_color)

		# ############################
		# Detect Background Color
		# ############################

		colors_count = {}		
		(channel_b, channel_g, channel_r) = cv2.split(image)
		channel_b = channel_b.flatten()
		channel_g = channel_g.flatten()
		channel_r = channel_r.flatten()

		for i in xrange(len(channel_b)):
			RGB = str(channel_r[i]) + "," + str(channel_g[i]) + "," + str(channel_b[i])
			if RGB in colors_count:
				colors_count[RGB] += 1
			else:
				colors_count[RGB] = 1
				
		# background_color = background_checker(colors_count)
		color = {}
		colors = Counter(colors_count)
		most_color_in = colors.most_common(1)[0][0]
		most_color_rgb = most_color_in.split(",")
		most_color_R = most_color_rgb[0]
		most_color_G = most_color_rgb[1]
		most_color_B = most_color_rgb[2]

		color["most_color"]= most_color_in

		# print most_color_R, most_color_G, most_color_B

		# RGB code of Gray color is (128,128,128)
		if int(most_color_R) > 127 and int(most_color_G) > 127 and int(most_color_B) > 127:
			color["is_preferable_background"] = 1
		else:
			color["is_preferable_background"] = 0
			
		background_color = color

		# Write Temporary Uploaded File
		cv2.imwrite('result/result.jpg', image)

		# Criteria 1
		if time_taken_checker(imagePath) == True:	
			is_date_creation_found = True
			if time_taken_less_than_six_months(imagePath) == True:
				is_less_than_six_months = True
			else:
				is_less_than_six_months = False
		else:
			is_less_than_six_months = False

		# Criteria 2.1
		if total_blur > 100:
			is_not_blurry = True
		else:
			is_not_blurry = False
			
		# Criteria 2.2
		if total_red_eyes > 0:
			is_red_eye_found = True
		else:
			is_red_eye_found = False

		# Criteria 2.3
		# if total_text > 0:
			# is_text_found = True
		# else:
			# is_text_found = False

		# Criteria 3
		if total_faces == 1:
			if total_upperbodies == 0:
				is_head_or_upperbody_found = True
			else:
				is_head_or_upperbody_found = True
		else:
			is_head_or_upperbody_found = False

		# Criteria 4
		if total_eyes >= 2 and total_noses >= 1 and total_mouths >= 0:
			is_face_looking_at_camera = True
		else:
			is_face_looking_at_camera = False

		# Criteria 5
		if background_color["is_preferable_background"] == 1:
			is_preferable_background = True
		else:
			is_preferable_background = False

		# Conclusion

		passed = (is_less_than_six_months == True and is_not_blurry == True and is_red_eye_found == False and is_head_or_upperbody_found == True and is_face_looking_at_camera == True and is_preferable_background == True)

		# update returned data
		data.update({
			"passes": passed,
			"desc": {
					"is_date_creation_found": is_date_creation_found,
					"is_less_than_six_months": is_less_than_six_months,
					"is_not_blurry": is_not_blurry,
					"is_red_eye_found": is_red_eye_found,
					"is_head_or_upperbody_found": is_head_or_upperbody_found,
					"is_face_looking_at_camera": is_face_looking_at_camera,
					"is_preferable_background": is_preferable_background,
				}
			})
	
	# return a JSON response
	return JsonResponse(data)

# Functions

def get_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
 
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()
 
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
 
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image

def months_between(date1,date2):
    if date1>date2:
        date1,date2=date2,date1
    m1=date1.year*12+date1.month
    m2=date2.year*12+date2.month
    months=m2-m1
    if date1.day>date2.day:
        months-=1
    elif date1.day==date2.day:
        seconds1=date1.hour*3600+date1.minute+date1.second
        seconds2=date2.hour*3600+date2.minute+date2.second
        if seconds1>seconds2:
            months-=1
    return months

def time_taken_checker(image):
	try:
		info = image._getexif()
		time_taken = info[36867].replace('\x00', '') # get tag value & remove null bytes
		if time_taken:
			return True
		else:
			return False
	except Exception as e:		
		return False
	 
def time_taken_less_than_six_months(image):
	# exif = exifread.process_file(imagePath)
	# dt = str(exif['exif datetimeoriginal'])  # might be different
	# # segment string dt into date and time
	# day, dtime = dt.split(" ", 1)
	# # segment time into hour, minute, second
	# hour, minute, second = dtime.split(":", 2)
	# return dtime
	try:
		info = image._getexif()
		str_time_taken = info[36867].replace('\x00', '') # get tag value & remove null bytes
		date_time_taken = dt.datetime.strptime(str_time_taken, '%Y:%m:%d %H:%M:%S')
		date_time_now = dt.datetime.now()
		month_difference = months_between(date_time_taken, date_time_now)
		if month_difference < 6:
			return True
		else:
			return False
	except Exception as e:		
		return False

# def text_checker(image):
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 	gray = cv2.medianBlur(gray, 3)

# 	filename = "{}.png".format(os.getpid())
# 	cv2.imwrite(filename, gray)

# 	text = pytesseract.image_to_string(Image.open(filename))
# 	os.remove(filename)
# 	print text

# 	return len(text) > 0

def text_checker(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale
	_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # threshold
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
	contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

	idx =0
	# for each contour found, draw a rectangle around it on original image
	for contour in contours:

		# get rectangle bounding contour
		[x,y,w,h] = cv2.boundingRect(contour)

		# discard areas that are too large
		if h>300 and w>300:
			continue

		# discard areas that are too small
		if h<40 or w<40:
			continue

		# draw rectangle around contour on original image
		cv2.rectangle(image,(x,y),(x+w,y+h), (255,0,255), 2)

		idx += 1

		roi = image[y:y + h, x:x + w]

		cv2.imwrite('results/result' + str(idx) + '.jpg', roi)

	if idx>0:
		return True
	else:
		return False

# def text_checker(image):
# 	vis      = image.copy()

# 	# Extract channels to be processed individually
# 	channels = cv2.text.computeNMChannels(image)
# 	# Append negative channels to detect ER- (bright regions over dark background)
# 	cn = len(channels)-1
# 	for c in range(0,cn):
# 		channels.append((255-channels[c]))

# 	# Apply the default cascade classifier to each independent channel (could be done in parallel)
# 	for channel in channels:

# 		erc1 = cv2.text.loadClassifierNM1('classifiers/trained_classifierNM1.xml')
# 		er1 = cv2.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)

# 		erc2 = cv2.text.loadClassifierNM2('classifiers/trained_classifierNM2.xml')
# 		er2 = cv2.text.createERFilterNM2(erc2,0.5)

# 		regions = cv2.text.detectRegions(channel,er1,er2)

# 		rects = cv2.text.erGrouping(image,channel,[r.tolist() for r in regions])
# 		#rects = cv.text.erGrouping(image,channel,[x.tolist() for x in regions], cv.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

# 		#Visualization
# 		for r in range(0, np.shape(rects)[0]):
# 			rect = rects[r]
# 			cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
# 			cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
# 			cv2.imwrite('results/result_text.jpg', vis)

# 			return True

def red_eye_checker(roi_color, ex, ey, ew, eh, part):
	eye_color = roi_color[ey:ey+eh, ex:ex+ew]
	eye_blurred = cv2.GaussianBlur(eye_color, (11, 11), 0)
	eye_hsv = cv2.cvtColor(eye_blurred, cv2.COLOR_BGR2HSV)

	# ############################
	# Red Eye Detection
	# ############################

	lower = (166, 84, 141)
	upper = (186, 255, 255)

	kernel = np.ones((9, 9), np.uint8)
	mask = cv2.inRange(eye_hsv, lower, upper)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size. Correct this value for your obect's size
		if radius > 0.5:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(eye_color, (int(x), int(y)), int(radius + 1), (0,255,255), 1)
			cv2.imwrite('results/result_face_right_red_' + part + '_eye.jpg', eye_color)

			return True
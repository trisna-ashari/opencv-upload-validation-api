#!/bin/bash

# Install OpenCV for this project
sudo apt update
sudo apt upgrade
sudo apt install python-opencv -y
sudo apt install python-pip -y

# Install libraries
pip install exifread pillow numpy django requests djangorestframework markdown django-filter

# Run migration
python manage.py migrate

# Open port 8000
sudo ufw allow 8000

# Start server
python manage.py runserver 0.0.0.0:8000

# Test by CURL
curl -X POST 'http://localhost:8000/validation/validate/' -F "image=@samples/a.jpg"

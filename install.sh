#!/bin/bash
# Symbol Vars
CHECK='[OK]'
CROSS='[NOT OK]'
INFO='[INFO]'
WARNING='[WARNING]'
SKIPPED='[SKIPPED]'

# Color Vars
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Install OpenCV for this project
printf "\n#${YELLOW} Installing OpenCV Python:${NC}\n"
printf "* Get apt update...\n"
sudo apt update
printf "* Get apt upgrade...\n"
sudo apt upgrade
printf "* Installing python-opencv...\n"
sudo apt install python-opencv -y
printf "* Installing python-pip...\n"
sudo apt install python-pip -y
printf "${GREEN}${CHECK} OpenCV Python has been successfully installed${NC}\n"

# Install libraries
printf "\n#${YELLOW} Installing libraries:${NC}\n"
printf "* Installing install exifread pillow numpy django requests djangorestframework markdown django-filter...\n"
pip install exifread pillow numpy django requests djangorestframework markdown django-filter
printf "${GREEN}${CHECK} Libraries has been successfully installed${NC}\n"

# Run migration
printf "\n#${YELLOW} Running migration:${NC}\n"
python manage.py migrate

# Open port 8000
printf "\n#${YELLOW} Open PORT 8000:${NC}\n"
sudo ufw allow 8000

# Start server
printf "\n#${YELLOW} Start server at PORT 8000:${NC}\n"
python manage.py runserver 0.0.0.0:8000

# Test by CURL
printf "\n#${YELLOW} Running test:${NC}\n"
curl -X POST 'http://localhost:8000/validation/validate/' -F "image=@samples/a.jpg"

cat <<"EOF"

    Powered By:
     ____________  __________  __  ___________  __    __  __    __  __
    /____  _____/ / ______  / / / / _________/ /  |  / / / /   / / / /
        / /      / /_____/ / / / / /________  /   | / / / /   / / / /
       / /      / ___   __/ / / /_______   / / /| |/ / / /   / / / /
      / /      / /   | |   / / ________ / / / / |   / / /___/ / / /___
     /_/      / /    | |  /_/ /__________/ /_/  |__/ /_______/ /_____/

    ##################################################################

EOF
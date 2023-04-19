#!/usr/bin/env python3

import cv2
import filters
import numpy as np

from webcam import Webcam
from dotenv import dotenv_values
from cvlib import convolve

env = dotenv_values(".env")
uname = env["uname"]
passwd = env["passwd"]
ip = env["ip"]
port = env["port"]

url = f"https://{uname}:{passwd}@{ip}:{port}/video"

webcam = Webcam(url)

window = cv2.namedWindow("window")

gx, gy = filters.gaussian_sep(15, 4)

for img in webcam:
    img = filters.convolve_sep(img, gx, gy, 1)

    cv2.imshow("window", img)

    match cv2.pollKey():
        case 0x71:  # q
            webcam.close()
            break

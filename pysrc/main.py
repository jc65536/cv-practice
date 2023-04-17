#!/usr/bin/env python3

import cv2
import filters

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

filter = filters.gaussian(11, 20)
filter = filters.grad_y

for img in webcam:
    img = convolve(img, filters.grad_xy) * 4

    cv2.imshow("window", img)

    match cv2.pollKey():
        case 0x71:  # q
            webcam.close()
            break

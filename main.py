#!/usr/bin/env python3

import cv2

from webcam import Webcam
from dotenv import dotenv_values
from filters import *

env = dotenv_values(".env")
uname = env["uname"]
passwd = env["passwd"]
ip = env["ip"]
port = env["port"]

url = f"https://{uname}:{passwd}@{ip}:{port}/video"

webcam = Webcam(url)

window = cv2.namedWindow("window")

filter = gaussian_filter(25, 10)
print(filter)

for img in webcam:
    img = convolve(img, filter)

    cv2.imshow("window", img)

    while True:
        match cv2.pollKey():
            case 0x71:  # q
                webcam.close()
                break

            case 0x6e: # n
                break

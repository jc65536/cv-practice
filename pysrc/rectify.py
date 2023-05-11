#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageDraw

green = (0, 255, 0)

ctrl_pts = ((252, 150),
            (934, 148),
            (1050, 653),
            (148, 655))

out_size = 800

dest_pts = ((0, 0),
            (out_size, 0),
            (out_size, out_size),
            (0, out_size))


def to_homo(v):
    x, y = v
    return np.array((x, y, 1))


def to_hetero(v):
    return (v / v[-1])[:-1]


def draw_dot(d: ImageDraw.ImageDraw, p: tuple[int, int]):
    r = 4
    x, y = p
    bbox = ((x - r, y - r), (x + r, y + r))
    d.ellipse(bbox, fill=green)


with Image.open("img/chessboard.jpg") as img:
    draw = ImageDraw.Draw(img)

    for p in ctrl_pts:
        draw_dot(draw, p)

    A = np.concatenate([[[-x, -y, -1,  0,  0,  0, x * xt, y * xt, xt],
                         [0,  0,  0, -x, -y, -1, x * yt, y * yt, yt]]
                        for (x, y), (xt, yt) in zip(ctrl_pts, dest_pts)])

    print("==== A ====")
    print(A)

    _, _, vh = np.linalg.svd(A)

    min_eig = vh[-1]
    min_eig /= min_eig[-1]
    H = min_eig.reshape((3, 3))
    print("==== H ====")
    print(H)

    print("==== Testing H ====")

    for p in ctrl_pts:
        p = to_homo(p)
        pt = np.matmul(H, p)
        pt = to_hetero(pt)
        print(pt)

    Hi = np.linalg.inv(H)
    print("==== H inv. ====")
    print(Hi)

    print("=== Testing inverse ====")
    print(np.matmul(H, Hi))

    out_arr = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    for i in range(0, out_size):
        for j in range(0, out_size):
            coord_dest = to_homo((j, i))
            coord_src = np.matmul(Hi, coord_dest)
            xs, ys = to_hetero(coord_src)
            # draw.point((xs, ys))
            try:
                out_arr[i, j, :] = img.getpixel((xs, ys))
            except IndexError:
                print(f"i: {i}; j: {j}; xs: {xs}; ys: {ys}")

    out = Image.fromarray(out_arr, "RGB")

    out.show()
    img.show()


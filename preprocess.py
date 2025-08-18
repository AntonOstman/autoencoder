import cv2 as cv
import numpy as np
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


im_nr = 9999

folders = ['Cat', 'Dog']

for folder in folders:
    processed_path = f"data/Processed/{folder}"
    data_path = f"data/PetImages/{folder}"

    if not os.path.isdir(processed_path):
        os.makedirs(processed_path)
        print(f"Created {processed_path}")

    failed_imgs = 0
    not_found_imgs = 0
    good_imgs = 0
    print(f"Reading {folder} folder")
    for i in range(im_nr):

        img = cv.imread(f"{data_path}/{i}.jpg")

        if img is None:
            not_found_imgs += 1
            continue

        temp_shape = img.shape
        if i % 3333 == 0:
            print(f"{i / im_nr:0.2f}")


        imsize = 250
        if (img.shape[0] > imsize and img.shape[1] > imsize):

            resized_image = cv.resize(img, (80,80), interpolation = cv.INTER_CUBIC)
            assert(resized_image.shape == [80, 80, 3])

            breakpoint()

            cv.imwrite(f"{processed_path}/{i}.jpg", resized_image)

            good_imgs += 1

        else:
            failed_imgs += 1


    print(f"{folder} failed imgs {failed_imgs} good imgs {good_imgs} not found imgs {not_found_imgs}")

# opticalFlow v1.0 | 2020 | EPOCH Foundation
# This work is licensed under the GNU GENERAL PUBLIC LICENSE

# Project ID: 8yGuHp7y

# Official configs for EPOCH_opticalFlow.

#----------------------------------------------------------------------------

import numpy as np
import cv2 as cv
from PIL import Image

#----------------------------------------------------------------------------

source = "/datasets/input.mp4"
OUTDIR = "/opticalFlow/render/v001"

#----------------------------------------------------------------------------

cap = cv.VideoCapture(source)
ret, image = cap.read()

image = image[:,:,::-1]

rgb_weights = [0.2989, 0.5870, 0.1140]
prev_frame = np.dot(image[...,:3], rgb_weights)

mask = np.zeros_like(image)
mask[..., 1] = 255

#----------------------------------------------------------------------------

count = 0
while(cap.isOpened()):

    # Read sequence of frames and convert to grayscale.
    ret, frame = cap.read()
    frame = np.dot(frame[...,:3], rgb_weights)

    # Display progress
    print("Processing frame:", count)

    # Calculate Dense Optical Flow between previous and current frames using Farneback method.
    flow = cv.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes magnitude and angle of 2D vectors.
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image value according to optical flow direction.
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # Save processed frame.
    img = Image.fromarray(rgb, 'RGB')
    img.save(OUTDIR + '/output_oflow_{:04d}.png'.format(count))

    # Update previous frame.
    prev_frame = frame

    count += 1

# Close all windows and free up resources.
cap.release()
cv.destroyAllWindows()

# opticalFlow
An implementation of the Farneback approach for dense motion estimation of an image sequence.

Dense optical flow attempts to compute the optical flow vector for every pixel of each frame. While such computation may be slower, it gives a more accurate result and a denser result suitable for applications such as learning structure from motion and video segmentation. There are various implementations of dense optical flow. We will be using the Farneback method, one of the most popular implementations, with using OpenCV, an open source library of computer vision algorithms, for implementation.


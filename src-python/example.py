#!/usr/bin/env python3

"""
This is EXAMPLE python code showing how to use the Python darknet module.
It is NOT a final application!  You must modify this example to make it do what you need.
"""

import os
import cv2
#import random
import darknet

# For this example, we're going to use the MS COCO dataset trained using YOLOv4-tiny.
# The .cfg and .names file are in the repo, and the weights can be downloaded from here:
#
#     https://github.com/hank-ai/darknet#mscoco-pre-trained-weights
#
cfg_file = "../cfg/yolov4-tiny.cfg"
names_file = "../cfg/coco.names"
weights_file = "../yolov4-tiny.weights"

# First thing we do is load the neural network.
network = darknet.load_net_custom(cfg_file.encode("ascii"), weights_file.encode("ascii"), 0, 1)
class_names = open(names_file).read().splitlines()

# Generate some random colours to use for each class.  If you don't want the colours to be random,
# then set the seed to a hard-coded value.
#random.seed(3)
colours = darknet.class_colors(class_names)

prediction_threshold = 0.5
width = darknet.network_width(network)
height = darknet.network_height(network)

# Iterate over several sample images in the repo's "artwork" directory.
for filename in ["../artwork/dog.jpg", "../artwork/eagle.jpg", "../artwork/giraffe.jpg", "../artwork/horses.jpg", "../artwork/person.jpg", "../artwork/scream.jpg"]:
    print(filename)

    # use OpenCV to load the image and swap OpenCV's usual BGR for the RGB that Darknet requires
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # create a Darknet-specific image structure with the resized image
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    # this is where darknet is called to do the magic!
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=prediction_threshold)
    darknet.free_image(darknet_image)

    # display the results on the console
    darknet.print_detections(detections, True)

    # draw some boxes and labels over what was detected
    image_with_boxes = darknet.draw_boxes(detections, image_resized, colours)

    cv2.imshow("annotated image", cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    if cv2.waitKey() & 0xFF == ord('q'):
        break;

darknet.free_network_ptr(network)

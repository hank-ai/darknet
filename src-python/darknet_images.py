#!/usr/bin/env python3

import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet




# Define a function named 'parser' to create and configure an argument parser
def parser():
    # Create an ArgumentParser object with a description
    parser = argparse.ArgumentParser(description="YOLO Object Detection")

    # Define command-line arguments and their descriptions
    # The default values are provided for each argument.

    # --input: Specifies the source of the images (single image, txt file with image paths, or folder)
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a txt with paths to them, or a folder. Image valid formats are jpg, jpeg, or png.")

    # --batch_size: Specifies the number of images to process at the same time
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")

    # --weights: Specifies the path to the YOLO weights file
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")

    # --dont_show: If provided, prevents displaying inference results in a window (useful for headless systems)
    parser.add_argument("--dont_show", action='store_true',
                        help="window inference display. For headless systems")

    # --ext_output: If provided, displays bounding box coordinates of detected objects
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")

    # --save_labels: If provided, saves detections' bounding box coordinates in YOLO format for each image
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")

    # --config_file: Specifies the path to the YOLO configuration file
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")

    # --data_file: Specifies the path to the YOLO data file
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")

    # --thresh: Sets the confidence threshold for removing detections with lower confidence
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")

    # --gpu: If provided, indicates the use of GPU for processing
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for processing")

    # --nms_thresh: Specifies the Non-Maximum Suppression threshold. If set, applies NMS.
    parser.add_argument("--nms_thresh", type=float, default=None,
                        help="Non-Maximum Suppression threshold. If set, applies NMS.")

    # Parse the command-line arguments and return the result
    return parser.parse_args()

# This function defines the command-line arguments and their descriptions,
# and it returns the parsed arguments when called.


# Define a function named 'check_arguments_errors' that takes 'args' as input
def check_arguments_errors(args):
    # Check if the threshold value is within the valid range (0 < thresh < 1)
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"

    # Check if the specified YOLO configuration file exists
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))

    # Check if the specified YOLO weights file exists
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))

    # Check if the specified YOLO data file exists
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))

    # If '--input' is provided, check if the specified input image or file exists
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))

# This function performs several checks on the command-line arguments and raises ValueError
# with appropriate error messages if any of the checks fail.


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


# Define a function named 'prepare_batch' that takes 'images', 'network', and an optional 'channels' argument
def prepare_batch(images, network, channels=3):
    # Get the width and height of the Darknet network
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    # Initialize an empty list to store the processed images
    darknet_images = []

    # Iterate through each input image
    for image in images:
        # Convert the input image from BGR to RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the dimensions of the Darknet network
        image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

        # Transpose the image to match the Darknet network's channel order (H x W x C to C x H x W)
        custom_image = image_resized.transpose(2, 0, 1)

        # Append the processed image to the list
        darknet_images.append(custom_image)

    # Concatenate the processed images into a single numpy array
    batch_array = np.concatenate(darknet_images, axis=0)

    # Normalize the pixel values to the range [0, 1] and convert to a contiguous array
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32) / 255.0

    # Get a pointer to the data as a ctypes float pointer
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))

    # Create a Darknet IMAGE object with the batched data and return it
    return darknet.IMAGE(width, height, channels, darknet_images)

# This function takes a list of input images, prepares them for Darknet inference by resizing,
# reordering channels, and normalizing pixel values, and returns a Darknet IMAGE object containing the batched data.



# Define a function named 'image_detection' that takes several arguments
def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Get the width and height of the Darknet network
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    # Create a Darknet IMAGE object with the specified width, height, and 3 channels
    darknet_image = darknet.make_image(width, height, 3)

    # Check if 'image_or_path' is a path to an image file (string) or an image array
    if isinstance(image_or_path, str):
        # Load the image from the provided file path
        image = cv2.imread(image_or_path)

        # Check if the image loading was successful
        if image is None:
            raise ValueError(f"Unable to load image {image_or_path}")
    else:
        # Use the provided image array
        image = image_or_path

    # Convert the input image from BGR to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the dimensions of the Darknet network
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # Copy the resized image data into the Darknet IMAGE object
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    # Perform object detection on the image using Darknet
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

    # Free the memory used by the Darknet IMAGE object
    darknet.free_image(darknet_image)

    # Draw bounding boxes and labels on the image based on the detected objects
    image_with_boxes = darknet.draw_boxes(detections, image_resized, class_colors)

    # Convert the image back to BGR color space (OpenCV format) and return it along with detections
    return cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), detections

# This function takes an image (either as a file path or an image array), a Darknet network,
# class names, class colors, and a detection threshold, and returns the image with bounding boxes
# and labels drawn around detected objects, as well as the list of detections.




# Define a function named 'batch_detection' that takes several arguments
def batch_detection(network, images, class_names, class_colors, thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    # Get the height and width of the images in the batch and check their shape
    image_height, image_width, _ = check_batch_shape(images, batch_size)

    # Prepare the batch of images for Darknet inference
    darknet_images = prepare_batch(images, network)

    # Perform batch inference on the prepared images using Darknet
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)

    # Initialize a list to store batch predictions
    batch_predictions = []

    # Iterate over the batched detections
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets

        # Apply Non-Maximum Suppression (NMS) if nms is specified
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)

        # Remove negative detections and get the final predictions
        predictions = darknet.remove_negatives(detections, class_names, num)

        # Draw bounding boxes and labels on the images based on the detected objects
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)

        # Append the predictions for this image to the batch predictions list
        batch_predictions.append(predictions)

    # Free the memory used by the batch detections
    darknet.free_batch_detections(batch_detections, batch_size)

    # Return the images with bounding boxes and labels drawn and the batch predictions
    return images, batch_predictions

# This function takes a Darknet network, a batch of images, class names, class colors, and
# several optional parameters for object detection, performs batch inference, and returns the
# images with bounding boxes and labels drawn and a list of batch predictions.



# Define a function named 'image_classification' that takes an image, a Darknet network, and class names
def image_classification(image, network, class_names):
    # Get the width and height of the Darknet network
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    # Convert the input image from BGR to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the dimensions of the Darknet network
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # Create a Darknet IMAGE object with the specified width, height, and 3 channels
    darknet_image = darknet.make_image(width, height, 3)

    # Copy the resized image data into the Darknet IMAGE object
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    # Perform image classification on the input image using Darknet
    detections = darknet.predict_image(network, darknet_image)

    # Create a list of (class_name, prediction_score) tuples for each class
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]

    # Free the memory used by the Darknet IMAGE object
    darknet.free_image(darknet_image)

    # Sort the predictions in descending order of prediction scores
    return sorted(predictions, key=lambda x: -x[1])

# This function takes an input image, performs image classification using a Darknet network,
# and returns a list of class predictions along with their associated scores.


# Function to convert bounding box coordinates to relative format
def convert2relative(image, bbox):
    """
    YOLO format uses normalized coordinates for annotation.

    Args:
        image: Input image (numpy array).
        bbox: Bounding box in absolute coordinates (x, y, width, height).

    Returns:
        Tuple representing bounding box coordinates in relative format (x_rel, y_rel, w_rel, h_rel).
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

# Function to save object detection annotations in YOLO format
def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates.

    Args:
        name: Name of the input image file.
        image: Input image (numpy array).
        detections: List of detected objects, each represented as (label, confidence, bbox).
        class_names: List of class names.

    Saves:
        Text file with YOLO-style annotations for object detection.
    """
    # Determine the output file name based on the input image name
    file_name = os.path.splitext(name)[0] + ".txt"

    # Open the output file for writing
    with open(file_name, "w") as f:
        # Iterate through detected objects
        for label, confidence, bbox in detections:
            # Convert bounding box coordinates to relative format
            x, y, w, h = convert2relative(image, bbox)

            # Find the index of the class label in class_names
            label = class_names.index(label)

            # Write annotation in YOLO format to the text file
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))



# Function to demonstrate batch object detection
def batch_detection_example():
    # Parse command line arguments using the 'parser' function
    args = parser()

    # Check for errors in the provided arguments
    check_arguments_errors(args)

    # Define the batch size for object detection
    batch_size = 3

    # Set a deterministic seed for random bbox colors
    random.seed(3)

    # Load the Darknet network, class names, and class colors from configuration files
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )

    # List of image file names for batch detection
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']

    # Read the images from file
    images = [cv2.imread(image) for image in image_names]

    # Perform batch object detection on the images
    images, detections = batch_detection(network, images, class_names,
                                         class_colors, batch_size=batch_size)

    # Save the detected images with bounding boxes
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)

    # Print the detections
    print(detections)


# Function to perform object detection on images
def perform_detection(args, network, class_names, class_colors):
    # Load a list of image paths from the input directory
    images_paths = load_images(args.input)

    # Iterate over the image paths and perform object detection
    for image_path in images_paths:
        prev_time = time.time()

        # Perform image detection on the current image
        image, detections = image_detection(image_path, network, class_names, class_colors, args.thresh)

        # Print detections and calculate frames per second (FPS)
        darknet.print_detections(detections, args.ext_output)
        fps = int(1 / (time.time() - prev_time))
        print("FPS: {}".format(fps))

        # Save annotations in YOLO format if requested
        if args.save_labels:
            save_annotations(image_path, image, detections, class_names)

        # Display the image with detections unless 'dont_show' is enabled
        if not args.dont_show:
            cv2.imshow('Inference', image)

            # Exit on 'q' key press
            if cv2.waitKey() & 0xFF == ord('q'):
                break


# Main function to handle object detection
def main():
    # Parse command line arguments using the 'parser' function
    args = parser()

    # Check for errors in the provided arguments
    check_arguments_errors(args)

    # If GPU is specified, set the GPU to use (GPU index 0 in this example)
    if args.gpu:
        darknet.set_gpu(0)

    # Load the Darknet network, class names, and class colors from configuration files
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    # Perform object detection on images using the loaded network and settings
    perform_detection(args, network, class_names, class_colors)

# Entry point of the script
if __name__ == "__main__":
    main()


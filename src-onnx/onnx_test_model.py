#!/bin/python3

# Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
# Copyright 2025 Stephane Charette
#
# Simple test inference using a .onnx file and an image.
#
# WARNING:  This code was generated using chatgpt.


import sys
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if len(sys.argv) != 3:
    print("Must specify one .onnx filename and the image filename.")
    exit(1)

# load the neural network
session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 160, 224]
output_names = [o.name for o in session.get_outputs()]

print(f"Input name:   {input_name}")
print(f"Input shape:  {input_shape}")
print(f"Output names: {output_names}")

# load the image
img = Image.open(sys.argv[2]).convert('RGB')
draw = ImageDraw.Draw(img)
W_img, H_img = img.size
_, _, H, W = input_shape
img_resized = img.resize((W, H))

# convert to either "float16" or "float32"; if darknet_onnx_export is called with -fp16, then the input image needs to be "float16"
img_np = np.array(img_resized).astype("float32") / 255.0
img_np = np.transpose(img_np, (2, 0, 1))  # (3, H, W)
img_np = np.expand_dims(img_np, axis=0)
print(f"Tensor shape: {img_np.shape}")

outputs = session.run(output_names, {input_name: img_np})

# Turn outputs into a dict for convenience
output_dict = {name: value for name, value in zip(output_names, outputs)}

confs = output_dict["confs"]    # (1, N, num_classes)
boxes = output_dict["boxes"]    # (1, N, 1, 4)

# Remove batch dimension
confs = confs[0]                # (N, num_classes)
boxes = boxes[0]                # (N, 1, 4)

# Squeeze the extra dimension on boxes: (N, 4)
boxes = np.squeeze(boxes, axis=1)  # now boxes.shape == (N, 4)

N, num_classes = confs.shape
print("boxes shape:", boxes.shape)  # (525, 4)
print("confs shape:", confs.shape)  # (525, 5)

# For each box, find best class and score
best_class_indices = np.argmax(confs, axis=1)              # (N,)
best_class_scores  = confs[np.arange(N), best_class_indices]  # (N,)

# Get top-k boxes by score
k = 10
topk_indices = np.argsort(-best_class_scores)[:k]

print(f"\nTop {k} detections (before NMS, just raw scores):")
for idx in topk_indices:
    score = best_class_scores[idx]
    cls   = best_class_indices[idx]
    x1, y1, x2, y2 = boxes[idx]  # (4,)
    print(f"  box {idx}: class={cls}, score={score:.4f}, "
          f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")

threshold = 0.75
detections = []

print(f"\nDetections with score >= {threshold}:")
for i in range(N):
    for c in range(num_classes):
        score = confs[i, c]
        if score >= threshold:
            x1, y1, x2, y2 = boxes[i]
            x1_pix = x1 * W_img
            y1_pix = y1 * H_img
            x2_pix = x2 * W_img
            y2_pix = y2 * H_img
            detections.append((score, c, (x1_pix, y1_pix, x2_pix, y2_pix)))
            print(f"  box {i}: class={c}, score={score:.4f}, "
                  f"box=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")

class_names = [f"class_{i}" for i in range(num_classes)]
detections.sort(key=lambda x: -x[0])
font = ImageFont.load_default()

for score, cls, (x1, y1, x2, y2) in detections:
    label = f"{class_names[cls]} {score:.2f}"
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
#    text_w, text_h = draw.textsize(label, font=font)
#    text_bg = [x1, y1 - text_h, x1 + text_w, y1]
#    draw.rectangle(text_bg, fill="red")
#    draw.text((x1, y1 - text_h), label, fill="white", font=font)
    bbox = draw.textbbox((x1, y1), label, font=font)  # (left, top, right, bottom)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Place label *above* the box, if room, otherwise inside
    text_x = x1
    text_y = max(0, y1 - text_h)

    text_bg = [text_x, text_y, text_x + text_w, text_y + text_h]

    # Draw filled rectangle for label background
    draw.rectangle(text_bg, fill="red")
    draw.text((text_x, text_y), label, fill="white", font=font)

img.show()

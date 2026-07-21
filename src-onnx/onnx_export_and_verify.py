#!/usr/bin/env python3

"""Export model to ONNX format and compare outputs to darknet.

This tool runs the darknet_onnx_export binary to export a Darknet model
to ONNX format, then runs inference using the darknet python bindings
and the ONNX runtime to verify that the outputs match. This is useful
for testing the implementation of the ONNX exporter.

This tool exits with status 0 if all detections match, and non-zero if
any detections do not match."""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import darknet


def export_onnx(cfg_file, weights_file, fp16=False):
    """Exports Darknet model to ONNX format and returns path to ONNX file."""
    extra = ["-fp16"] if fp16 else []
    subprocess.check_call(["darknet_onnx_export", cfg_file, weights_file] + extra)
    output = Path(cfg_file).with_suffix(".onnx")
    return output


def load_image(image_path, width, height):
    """Loads an image and resizes it to the given width and height."""
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(
        image_rgb, (width, height), interpolation=cv2.INTER_LINEAR
    )
    return image_resized


def darknet_detect(network, image, class_names, threshold):
    """Performs inference using the Darknet C library and returns detections."""
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image.tobytes())
    # NOTE: We disable NMS because the ONNX export does not currently do NMS.
    detections = darknet.detect_image(
        network, class_names, darknet_image, thresh=threshold, nms=False
    )
    darknet.free_image(darknet_image)
    return detections


def onnx_detect(session, image, class_names, threshold):
    """Performs inference using the ONNX model and returns detections."""
    dtype = np.float16 if "float16" in session.get_inputs()[0].type else np.float32

    img_h, img_w, _ = image.shape
    img_in = image.astype(dtype) / 255.0
    img_in = np.transpose(img_in, (2, 0, 1))  # (3, H, W)
    img_in = np.expand_dims(img_in, axis=0)
    confs, boxes = session.run(["confs", "boxes"], {"frame": img_in})
    confs = confs[0]  # (N, num_classes)
    boxes = boxes[0].squeeze(axis=1)  # (N, 4)

    # For each box, find best class and score
    N = confs.shape[0]
    best_class_indices = np.argmax(confs, axis=1)  # (N,)
    best_class_scores = confs[np.arange(N), best_class_indices]  # (N,)

    # Convert into same detections format as `detect_image`.
    outputs = []
    idx = np.nonzero(best_class_scores > threshold)[0]
    for i in idx:
        score = best_class_scores[i]
        cls = best_class_indices[i]
        x1, y1, x2, y2 = boxes[i].astype(np.float32)
        box = (
            (x1 + x2) / 2 * img_w,
            (y1 + y2) / 2 * img_h,
            (x2 - x1) * img_w,
            (y2 - y1) * img_h,
        )
        outputs.append((class_names[int(cls)], f"{score * 100:0.2f}", tuple(box)))
    return sorted(outputs, key=lambda v: v[1])


def single_detection_eq(a, b, atol_conf=0.15, atol_box=0.5):
    """Compares two detections and returns True if they are equivalent.

    Darknet and ONNX Runtime use different convolution/GEMM implementations, so their
    float32 summation order differs and results are not bit-identical even when both
    exporter and runtime are behaving correctly.  In practice this shows up as noise on
    the order of ~0.1 percentage points of confidence and ~0.1px of box position (fp32),
    and considerably more for fp16, since fp16 has far fewer significant bits to begin
    with and that quantization error compounds over ~100+ layers.  The default tolerances
    are set loosely enough to absorb fp32 numerical noise while still catching real (much
    larger) export bugs; pass larger tolerances for fp16 models."""
    (name_a, conf_a, box_a), (name_b, conf_b, box_b) = a, b
    return (
        name_a == name_b
        and np.allclose(float(conf_a), float(conf_b), atol=atol_conf)
        and np.allclose(box_a, box_b, atol=atol_box)
    )


def greedy_nearest_match(dets_a, dets_b, atol_conf, atol_box):
    """Greedily matches dets_a against dets_b, returning unmatched detections.

    Checks if detections are "equal" via `single_detection_eq`, however this is
    not enough to do the matching, because due to floating point differences
    there might be multiple candidates. To resolve this, take the nearest 
    detection by distance (computed from the detection box).

    Returns (unmatched_a, unmatched_b)."""
    order = sorted(range(len(dets_a)), key=lambda i: -float(dets_a[i][1]))
    unmatched_b = list(dets_b)

    unmatched_a = []
    for i in order:
        da = dets_a[i]
        best_idx, best_dist = None, None
        for j, db in enumerate(unmatched_b):
            if not single_detection_eq(da, db, atol_conf, atol_box):
                continue
            dist = sum((x - y) ** 2 for x, y in zip(da[2], db[2]))
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = j, dist
        if best_idx is None:
            unmatched_a.append(da)
        else:
            unmatched_b.pop(best_idx)

    return unmatched_a, unmatched_b


def detections_eq(dets_a, dets_b, atol_conf=0.15, atol_box=0.5, threshold=0.75):
    """Returns True if the two lists of detections are equivalent, ignoring order.

    Detections are first matched using `greedy_nearest_match()`, which is a
    needed to handle floating point differences in the confidence and box 
    coordinates.

    Any detection left unmatched after that is still allowed, but only if its
    confidence is within atol_conf of --threshold. This is to account for floating
    point noise that might put a detection either side of the threshold.

    This is O(N^2) so only suitable for small numbers of detections."""
    unmatched_a, unmatched_b = greedy_nearest_match(dets_a, dets_b, atol_conf, atol_box)

    threshold_pct = threshold * 100.0
    for _, conf, _ in unmatched_a + unmatched_b:
        if abs(float(conf) - threshold_pct) > atol_conf:
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cfg", help="Model .cfg file")
    parser.add_argument("weights", help="Model .weights file")
    parser.add_argument("images", nargs="+", help="Input images to test")
    parser.add_argument("--names", default="../cfg/coco.names", help="Class names file")
    parser.add_argument(
        "--threshold", default=0.75, type=float, help="Detection threshold"
    )
    parser.add_argument("--draw", action="store_true", help="Draw boxes on images")
    parser.add_argument(
        "--print", action="store_true", help="Print detections to console"
    )
    parser.add_argument("--fp16", action="store_true", help="Export ONNX model in FP16")
    args = parser.parse_args()

    dn_network = darknet.load_net_custom(
        args.cfg.encode("ascii"), args.weights.encode("ascii"), 0, 1
    )
    width = darknet.network_width(dn_network)
    height = darknet.network_height(dn_network)
    class_names = open(args.names).read().splitlines()
    class_colors = darknet.class_colors(class_names)

    onnx_model = export_onnx(args.cfg, args.weights, fp16=args.fp16)
    onnx_network = ort.InferenceSession(
        onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    all_eq = True
    for image_path in args.images:
        image = load_image(image_path, width, height)
        dn_dets = darknet_detect(dn_network, image, class_names, args.threshold)
        onnx_dets = onnx_detect(onnx_network, image, class_names, args.threshold)

        if args.print:
            print("\n")
            darknet.print_detections(dn_dets, True)
            darknet.print_detections(onnx_dets, True)
            print("")

        # fp16 has far fewer significant bits than fp32, so its quantization error
        # (compounded over ~100+ layers) needs a looser tolerance to avoid false positives.
        atol_conf, atol_box = (1.0, 3.0) if args.fp16 else (0.15, 0.5)
        dets_eq = detections_eq(dn_dets, onnx_dets, atol_conf, atol_box, args.threshold)
        all_eq = all_eq and dets_eq
        print(f"{image_path}: detections match -> {dets_eq}")

        if args.draw:
            dn_img = darknet.draw_boxes(dn_dets, image.copy(), class_colors)
            onnx_img = darknet.draw_boxes(onnx_dets, image.copy(), class_colors)

            cv2.imshow(
                "darknet -- onnx",
                cv2.hconcat(
                    [
                        cv2.cvtColor(dn_img, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(onnx_img, cv2.COLOR_RGB2BGR),
                    ]
                ),
            )

            if cv2.waitKey() & 0xFF == ord("q"):
                break

    # Exit with non-zero return status if any detections do not match.
    if not all_eq:
        raise SystemExit(1)

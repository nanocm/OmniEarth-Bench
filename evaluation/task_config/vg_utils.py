import base64
import io
import logging
import re
from collections import defaultdict

import numpy as np
from datasets import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = 10_00000000
# eval_logger = logging.getLogger("lmms-eval")
from loguru import logger as eval_logger

COCO_REC_METRICS = [
    "IoU",
    "ACC@0.1",
    "ACC@0.3",
    "ACC@0.5",
    "ACC@0.7",
    "ACC@0.9",
]


def decode_base64_to_image(base64_string):
    # print(len(base64_string))
    num_padding = 4 - (len(base64_string) % 4)
    if num_padding < 4:
        base64_string += "=" * num_padding
    try:
        image_data = base64.b64decode(base64_string)
    except Exception as e:
        print(f"Length of base64 string {len(base64_string)}")
        # eval_logger.error(f"Error decoding base64 image: {e}")
        import sys

        sys.exit(1)
    image = Image.open(io.BytesIO(image_data))
    # if image.mode in ("RGBA", "P"):
    #     image = image.convert("RGB")
    # if target_size > 0:
    #     image.thumbnail((target_size, target_size))
    return image


def refcoco_bbox_rec_preprocess_dataset(dataset: Dataset):
    return dataset


def refcoco_bbox_rec_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    img = doc["image"]
    if type(img) is str:
        img = Image.open(img)
    return [img.convert("RGB")]



def refcoco_bbox_rec_doc_to_text(doc, lmms_eval_specific_kwargs):
    # assert isinstance(doc["answer"], str), "Answer must be a string"
    return doc["question"]



def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    # Define the regex pattern to find the first instance of four floats within square brackets
    # TODO: add more patterns to support various formats
    # pattern1 [num, num, num, num]
    pattern = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"

    # Use re.search to find the first match of the pattern in the input string
    match = re.search(pattern, input_str)

    # If a match is found, convert the captured groups into a list of floats
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # pattern2 (num, num, num, num)
    pattern = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # pattern3 (num, num), (num, num)
    pattern = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\),\s*\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # If the input does not contain the pattern, return the null float sequence
    return [0, 0, 0, 0]


def refcoco_bbox_rec_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    pred = parse_float_sequence_within(pred)
    ann_id = doc["index"]
    data_dict = {
        "answer": doc["answer"],
        "pred": pred,
        "ann_id": ann_id,
        # "bbox": doc["answer"],
        "w": doc["image_width"],
        "h": doc["image_height"],
        **{
            k: doc[k]
            for k in [
                "L1-task",
                "L2-task",
                "L3-task",
                "L4-task",
            ]
        },
    }
    return {f"refcoco_{metric}": data_dict for metric in COCO_REC_METRICS}


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    try:
        iou = intersection_area / union_area
    except ZeroDivisionError:
        iou = 0.0

    return iou


def refcoco_bbox_rec_aggregation_result(results, metric):
    """
    Aggregate the results of the RefCOCO evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - dict: Dictionary containing the aggregated results for the specified metric.
    """
    scorers = {
        "IoU": compute_iou,
        "ACC@0.1": lambda iou: iou >= 0.1,
        "ACC@0.3": lambda iou: iou >= 0.3,
        "ACC@0.5": lambda iou: iou >= 0.5,
        "ACC@0.7": lambda iou: iou >= 0.7,
        "ACC@0.9": lambda iou: iou >= 0.9,
    }
    results_dict = {_metric: [] for _metric in scorers.keys()}
    metrics = dict()
    # for i in range(len(results)):
    # result = results[i]
    for result in results:
        l1, l2, l3, l4 = [
            result[key]
            for key in [
                "L1-task",
                "L2-task",
                "L3-task",
                "L4-task",
            ]
        ]
        ref = result["answer"]
        pred = result["pred"]
        w, h = result["w"], result["h"]
        preds = [
            pred,
            [pred[0] / w, pred[1] / h, pred[2] / w, pred[3] / h],
            [num / 1000 for num in pred],
            [num / 100 for num in pred],
        ]
        iou = max(compute_iou(ref, pred) for pred in preds)
        metrics.setdefault(l1, dict())
        metrics[l1].setdefault(l2, dict())
        metrics[l1][l2].setdefault(l3, dict())
        metrics[l1][l2][l3].setdefault(l4, defaultdict(int))
        metrics[l1][l2][l3][l4]["IoU"] += iou
        metrics[l1][l2][l3][l4]["cnt"] += 1
        for metric in ["ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9"]:
            metrics[l1][l2][l3][l4][metric] += scorers[metric](iou)

        # for compatibility
        if metric == "IoU":
            score = iou
        elif "ACC" in metric:
            score = scorers[metric](iou)
        else:
            # never used
            assert False, f"Unknown metric: {metric}"
        results_dict[metric].append(score)

    for l1, l1_vals in metrics.items():
        eval_logger.info("*" * 36 + f"{l1} (Level-1 Task Start)")
        for l2, l2_vals in l1_vals.items():
            eval_logger.info("+" * 24 + f"{l2} (Level-2 Start)")
            for l3, l3_vals in l2_vals.items():
                eval_logger.info("+" * 12 + f"{l3} (Level-3 Start)")
                for l4, l4_vals in l3_vals.items():
                    iou = l4_vals["IoU"] / l4_vals["cnt"]
                    acc1 = l4_vals["ACC@0.1"] / l4_vals["cnt"]
                    acc3 = l4_vals["ACC@0.3"] / l4_vals["cnt"]
                    acc5 = l4_vals["ACC@0.5"] / l4_vals["cnt"]
                    acc7 = l4_vals["ACC@0.7"] / l4_vals["cnt"]
                    acc9 = l4_vals["ACC@0.9"] / l4_vals["cnt"]

                    eval_logger.info(
                        "-" * 6 + "\t IoU " + "{:.4f}".format(iou) + f"\t{l4.capitalize()} ({l4_vals['cnt']} items)\n"
                    )
                    for metric in ["ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9"]:
                        val = l4_vals[metric] / l4_vals["cnt"]
                        eval_logger.info("-" * 6 + f"\t {metric} " + "{:.4f}".format(val))

    results_dict[metric] = sum(results_dict[metric]) / len(results_dict[metric])
    print(f"Aggregated {metric} score: {results_dict[metric]}")
    return results_dict[metric]


def refcoco_bbox_rec_iou(results):
    return refcoco_bbox_rec_aggregation_result(results, "IoU")


def refcoco_bbox_rec_acc01(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.1")


def refcoco_bbox_rec_acc03(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.3")


def refcoco_bbox_rec_acc05(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.5")


def refcoco_bbox_rec_acc07(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.7")


def refcoco_bbox_rec_acc09(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.9")

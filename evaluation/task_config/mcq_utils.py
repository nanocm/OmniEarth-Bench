import base64
import datetime
import io
import json
import os
import re
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

Image.MAX_IMAGE_PIXELS = 10_0000_0000


def decode_bytes_to_image(image_bytes):
    return Image.open(BytesIO(image_bytes))


def decode_base64_to_image(base64_string):
    num_padding = 4 - (len(base64_string) % 4)
    if num_padding < 4:
        base64_string += "=" * num_padding
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def mme_realworld_doc_to_visual(doc):
    imgs = []
    for img in doc["image"]:
        if type(img) is str:
            img = Image.open(img)
        imgs.append(img)
    return [img.convert("RGB") for img in imgs]


def mme_realworld_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option_prompt = "The choices are listed below:\n" + "\n".join(doc["multi-choice options"]) + "\n"
    options = ", ".join("ABCDEFGH"[: len(doc["multi-choice options"])])

    pre_prompt = ""
    if doc["question_type"] in ["Single Choice", "Chain-of-Thought"]:
        select_prompt = "Select the best answer for the multiple-choice question based on the image. Only respond with the letter corresponding to the correct answer ({options})."
    elif doc["question_type"] == "Multiple Choice":
        select_prompt = "Select the best answer(s) for the multiple-choice question based on the image. There may be more than one correct option. Only respond with the letter(s) corresponding to the correct answer(s) ({options}), with multiple choices separated by spaces."
    else:
        assert False, f"Unknown question type: {doc['question_type']}"
    # select_prompt = lmms_eval_specific_kwargs["select_prompt"]
    post_promt = f"\n{select_prompt}\nThe answer is:"
    post_promt = post_promt.format(options=options)
    question += pre_prompt + option_prompt + post_promt

    return question


def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]):
    if type(s) is dict:
        s = ""
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if not re.search("[ABCDE]", s):
        # if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.findall(r"\(([a-eA-E])\)", s)
    if len(matches) == 0:
        # matches = re.findall(r"(?:^|\s)([a-eA-E])(?:$|[\s,.])", s)
        matches = re.findall(r"(?:^|\s)?([a-eA-E])(?:$|[\s,.])?", s)
    if len(matches) == 0:
        matches = re.findall(r"[a-eA-E]", s)
    if len(matches) == 0:
        return ""
    else:
        matches = set(mat.upper() for mat in matches)
        return "".join(matches)


def mme_realworld_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme_realworld score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    unable_to_decide = "ABCDEFGH"[len(doc["multi-choice options"]) - 1]
    assert "unable to decide" in doc["multi-choice options"][-1].lower()
    data_dict = {
        "question_id": doc["index"],
        "pred_answer": pred_ans,
        "answer": doc["answer"],
        "unable_to_decide": unable_to_decide,
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

    return {"mme_realworld_score": data_dict}



def mme_realworld_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    metrics = dict()
    for i in range(len(results)):
        result = results[i]
        l1, l2, l3, l4 = [
            result[key]
            for key in [
                "L1-task",
                "L2-task",
                "L3-task",
                "L4-task",
            ]
        ]
        cnt = 1 if set(result["pred_answer"]) == set(result["answer"]) else 0
        metrics.setdefault(l1, dict())
        metrics[l1].setdefault(l2, dict())
        metrics[l1][l2].setdefault(l3, dict())
        metrics[l1][l2][l3].setdefault(l4, defaultdict(int))
        metrics[l1][l2][l3][l4]["true"] += cnt
        metrics[l1][l2][l3][l4]["false"] += 1 - cnt
        metrics[l1][l2][l3][l4]["unable_to_decide"] += result["pred_answer"] == result["unable_to_decide"]
    sum_all, succ_all = 0, 0
    for l1, l1_vals in metrics.items():
        eval_logger.info("*" * 36 + f"{l1} (Level-1 Task Start)")
        cnt_l1, cnt_l1_unable, sum_l1 = 0, 0, 0
        for l2, l2_vals in l1_vals.items():
            eval_logger.info("+" * 24 + f"{l2} (Level-2 Start)")
            cnt_l2, cnt_l2_unable, sum_l2 = 0, 0, 0
            for l3, l3_vals in l2_vals.items():
                eval_logger.info("+" * 12 + f"{l3} (Level-3 Start)")
                cnt_l3, cnt_l3_unable, sum_l3 = 0, 0, 0
                for l4, l4_vals in l3_vals.items():
                    truth_s = l4_vals["true"]
                    false_s = l4_vals["false"]
                    unable_to_decide_s = l4_vals["unable_to_decide"]
                    cnt_l3 += truth_s
                    cnt_l3_unable += unable_to_decide_s
                    sum_l3 += truth_s + false_s
                    acc = truth_s / (truth_s + false_s)
                    eval_logger.info(
                        "-" * 6
                        + "\t Acc "
                        + "{:.4f}".format(acc)
                        + f"\t Unable choice {unable_to_decide_s} \t{l4.capitalize()} ({l4_vals['false'] + l4_vals['true']} items)\n"
                    )
                if sum_l3 == 0:
                    acc_l3 = 0
                    cnt_l3_unable = 0
                else:
                    acc_l3 = cnt_l3 / sum_l3
                eval_logger.info(
                    "+" * 12
                    + "\t Acc "
                    + "{:.4f}".format(acc_l3)
                    + f"\t Unable choice {cnt_l3_unable} \t{l3.capitalize()} ({sum_l3} items)\n"
                )
                cnt_l2 += cnt_l3
                cnt_l2_unable += cnt_l3_unable
                sum_l2 += sum_l3
            if sum_l2 == 0:
                acc_l2 = 0
                cnt_l2_unable = 0
            else:
                acc_l2 = cnt_l2 / sum_l2
            eval_logger.info(
                "+" * 24
                + "\t Acc "
                + "{:.4f}".format(acc_l2)
                + f"\t Unable choice {cnt_l2_unable} \t{l2.capitalize()} ({sum_l2} items)\n"
            )
            cnt_l1 += cnt_l2
            cnt_l1_unable += cnt_l2_unable
            sum_l1 += sum_l2
        if sum_l1 == 0:
            acc_l1 = 0
            cnt_l1_unable = 0
        else:
            acc_l1 = cnt_l1 / sum_l1
        eval_logger.info(
            "*" * 36
            + "\t Acc "
            + "{:.4f}".format(acc_l1)
            + f"\t Unable choice {cnt_l1_unable} \t{l1.capitalize()} ({sum_l1} items)\n"
        )
        succ_all += cnt_l1
        sum_all += sum_l1
    eval_logger.info("*" * 36 + "Overall Acc " + "{:.4f}".format(succ_all / sum_all))
    return succ_all / sum_all

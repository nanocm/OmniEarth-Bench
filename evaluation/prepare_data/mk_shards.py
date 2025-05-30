import base64
import io
import json
import math
import random
import re
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value, concatenate_datasets
from PIL import Image as PILImage
from tqdm import tqdm

PILImage.MAX_IMAGE_PIXELS = 10_0000_0000


def load_json(fpath, is_json_line=False):
    res = []
    with open(fpath, "r", encoding="utf8") as f:
        if is_json_line:
            for line in f:
                res.append(json.loads(line))
        else:
            res = json.load(f)
    return res


def encode_image_to_base64(image_path):
    img = PILImage.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    encoded_string = base64.b64encode(buf.getvalue()).decode("utf-8")
    # with open(image_path, "rb") as image_file:
    #     # 将图片编码为Base64字符串
    #     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def encode_image_to_bytes(image_path, format="PNG"):
    with PILImage.open(image_path) as img:
        img_byte_arr = io.BytesIO()
        # if img.mode != "RGB" and format == 'JPEG':
        # img = img.convert("RGB")
        img.save(img_byte_arr, format=format)  # 根据需要调整格式
        return img_byte_arr.getvalue()


def data_convert_cot(item, img_base: Path = Path("./")):
    img_names = item["Images"]
    # item["bytes"] = [encode_image_to_base64(img_path) for img_path in images]
    options = [option.strip() for option in item["Answer Choices"]]
    CoT = []
    key_annotation_steps = {
        "solution1": None,
        "solution2": None,
        "solution3": None,
        "solution4": None,
    }
    reference_caption = []
    if "CoT" in item:
        CoT = item["CoT"]
        # captions = []
        logicals = []
        for idx, step in enumerate(CoT):
            assert "Step " in step
            step = step[len("Step x:") :].strip()
            if "is a photo" in step:
                # captions.append(step)
                reference_caption.append(step)
            else:
                logicals.append(step)
        logicals.append(f"The answer is {item['Ground Truth']}")
        key_annotation_steps = {
            "solution1": {
                "image_caption": [""],
                "logical_conclusion": logicals,
            },
        }
    res = {
        "index": int(item["Question_id"].split("/")[-1]),
        "question_type": item["Question Type"],
        "question": item["Text"],
        "multi-choice options": options,
        "answer": item["Ground Truth"],
        "image": [str((img_base / img_name).absolute()) for img_name in img_names],
        "CoT": CoT,
        # for cot compatibility
        "key_annotation_steps": key_annotation_steps,
        "reference_caption": reference_caption,
        "category": item["L2-task"],
        "subcategory": item["L4-task"],
        # for cot compatibility
        **{
            k: item[k]
            for k in [
                "L1-task",
                "L2-task",
                "L3-task",
                "L4-task",
                "Dataset",
            ]
        },
    }
    return res


def data_convert_mcq(item, img_base: Path = Path("./")):
    img_names = item["Images"]
    # item["bytes"] = [encode_image_to_base64(img_path) for img_path in images]
    options = [option.strip() for option in item["Answer Choices"]]
    res = {
        "index": int(item["Question_id"].split("/")[-1]),
        "question_type": item["Question Type"],
        "question": item["Text"],
        "multi-choice options": options,
        "answer": item["Ground Truth"],
        "image": [str((img_base / img_name).absolute()) for img_name in img_names],
        **{
            k: item[k]
            for k in [
                "L1-task",
                "L2-task",
                "L3-task",
                "L4-task",
                "Dataset",
            ]
        },
    }
    return res


def data_convert_vg(item, img_base: Path = Path("./")):
    assert len(item["Images"]) == 1, "VG only support single image"
    img_name = item["Images"][0]
    img_path = (img_base / img_name).absolute()
    x_min, y_min, x_max, y_max = [float(num) for num in re.findall(r"<([\d.]+)>", item["Ground Truth"])]
    img = PILImage.open(img_path)
    img_width, img_height = img.width, img.height
    bbox = [
        x_min / img_width,
        y_min / img_height,
        x_max / img_width,
        y_max / img_height,
    ]
    # item["bytes"] = [encode_image_to_base64(img_path) for img_path in images]
    res = {
        "index": int(item["Question_id"].split("/")[-1]),
        "question_type": item["Question Type"],
        "question": item["Text"],
        "answer": bbox,
        "image": str(img_path),
        "image_width": img_width,
        "image_height": img_height,
        # "bbox": bbox,
        **{
            k: item[k]
            for k in [
                "L1-task",
                "L2-task",
                "L3-task",
                "L4-task",
                "Dataset",
            ]
        },
    }
    return res



def save_block(data_block, question_type):
    raw_data = data_block
    # validate question id
    l1, l2, l3, l4, _dataset = [raw_data[0][key] for key in ["L1-task", "L2-task", "L3-task", "L4-task", "Dataset"]]
    try:
        for item in raw_data:
            _ = int(item["Question_id"].split("/")[-1])
    except ValueError:
        for idx, item in enumerate(raw_data):
            item["Question_id"] = f"{l4}/{idx}"
    # question_type = raw_data[0]["Question Type"]
    raw_dataset = Dataset.from_list(data_block)
    if question_type in ["Single Choice", "Multiple Choice"]:
        features = Features(
            {
                "index": Value("int32"),
                "question": Value("string"),
                "question_type": Value("string"),
                "multi-choice options": [Value("string")],
                "answer": Value("string"),
                "image": Sequence(Image()),
                "L1-task": Value("string"),
                "L2-task": Value("string"),
                "L3-task": Value("string"),
                "L4-task": Value("string"),
                "Dataset": Value("string"),
            }
        )
        data_convert = data_convert_mcq
        task_dir = Path("./mcq_shards") / "/".join([l1, l2, l3]).replace(" ", "_")
    elif question_type == "Visual Grounding":
        features = Features(
            {
                "index": Value("int32"),
                "question": Value("string"),
                "question_type": Value("string"),
                "answer": Sequence(Value("float")),
                "image": Image(),
                "image_width": Value("int32"),
                "image_height": Value("int32"),
                "L1-task": Value("string"),
                "L2-task": Value("string"),
                "L3-task": Value("string"),
                "L4-task": Value("string"),
                "Dataset": Value("string"),
            }
        )
        data_convert = data_convert_vg
        task_dir = Path("./vg_shards") / "/".join([l1, l2, l3]).replace(" ", "_")
    elif question_type == "Chain-of-Thought":
        features = Features(
            {
                "index": Value("int32"),
                "question": Value("string"),
                "question_type": Value("string"),
                "multi-choice options": [Value("string")],
                "answer": Value("string"),
                "CoT": Sequence(Value("string")),
                "key_annotation_steps": {
                    "solution1": {
                        "image_caption": Sequence(Value("string")),
                        "logical_conclusion": Sequence(Value("string")),
                    },
                },
                "reference_caption": Sequence(Value("string")),
                "category": Value("string"),
                "subcategory": Value("string"),
                "image": Sequence(Image()),
                "L1-task": Value("string"),
                "L2-task": Value("string"),
                "L3-task": Value("string"),
                "L4-task": Value("string"),
                "Dataset": Value("string"),
            }
        )
        data_convert = data_convert_cot
        task_dir = Path("./cot_shards") / "/".join([l1, l2, l3]).replace(" ", "_")
        print("chain of thought")
    else:
        assert False, f"Unhandled Question Type: {question_type}"
    dataset = raw_dataset.map(
        data_convert,
        num_proc=16,
        remove_columns=raw_dataset.column_names,
        features=features,
    )
    task_dir.mkdir(exist_ok=True, parents=True)
    task_name = "__".join([l4]).replace(" ", "_")
    dataset.to_parquet(task_dir / f"{task_name}.parquet")
    print("Saving shards complete: ", task_dir / f"{task_name}.parquet")


if __name__ == "__main__":
    all_tasks = load_json("./tasks.json")
    for question_type, tasks in all_tasks.items():
        for task in tasks:
            raw_data = load_json(Path('./jsons') / f"{task}.json")
            save_block(raw_data, question_type)

import json
from collections import defaultdict
from pathlib import Path

import yaml


def load_json(fpath, is_json_line=False):
    res = []
    with open(fpath, "r", encoding="utf8") as f:
        if is_json_line:
            for line in f:
                res.append(json.loads(line))
        else:
            res = json.load(f)
    return res


if __name__ == "__main__":
    all_tasks = load_json("../prepare_data/tasks.json")
    question_type2path = {
        "Single Choice": "../prepare_data/mcq_shards/",
        "Multiple Choice": "../prepare_data/mcq_shards/",
        "Visual Grounding": "../prepare_data/vg_shards/",
    }
    question_type2yaml = {
        "Single Choice": "en_mcq_template_yaml",
        "Multiple Choice": "en_mcq_template_yaml",
        "Visual Grounding": "en_vg_template_yaml",
    }
    lmms_tag_suffix = {
        "Single Choice": "",
        "Multiple Choice": "_multi",
        "Visual Grounding": "_vg",
    }
    for question_type, tasks in all_tasks.items():
        if "Chain-of-Thought" == question_type:
            continue
        assert question_type in question_type2path, f"Unknown question type: {question_type}"
        l2 = defaultdict(list)
        for task in tasks:
            _TAG = "/".join(task.split("/")[:2]) + lmms_tag_suffix[question_type]
            parquet_path = Path(question_type2path[question_type] + task + ".parquet")
            assert parquet_path.exists(), "Error: parquet file not found: " + str(parquet_path)
            l2[_TAG].append(str(parquet_path.resolve()))
        for k, v in l2.items():
            task = k.replace("/", "-")
            tag = k.split("/")[0] + lmms_tag_suffix[question_type]
            v.sort()
            doc = {
                "dataset_kwargs": {"data_files": v},
                "task": task,
                "tag": tag,
                "include": question_type2yaml[question_type],
            }
            with open(f"{task}.yaml", "w", encoding="utf8") as f:
                content = yaml.safe_dump(doc)
                f.write(content)

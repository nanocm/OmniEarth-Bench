# Evaluation

## Prepare data

* First download the dataset from [huggingface](https://huggingface.co/datasets/initiacms/OmniEarth-Bench).
* Unzip raw.tar, and copy `jsons/` and `raw/` into the `prepare_data/`
* Run `mk_shards.py`. This will prepare the parquet files used in evaluation.

### Prepare task config

* cd into the `task_config/` and run `mk_yaml.py`. This will creates a bunch of yaml files, each of which stands for a L2 task.

  Be aware, to evaluate the cot tasks, you need to manually change the path in `cot.yaml`.

* Install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval?tab=readme-ov-file#installation) and copy `task_config/` into `lmms_eval/tasks/`

### Benchmark

To test on L1 task `Atmosphere`, for example, run the following command:

```bash
#!/usr/bin/bash
TASKS="Atmosphere"	# A tag, can also be Biosphere, Pedosphere, etc.
MODEL="qwen2_5_vl"
PRETRAINED_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ARGS="pretrained=${PRETRAINED_MODEL},use_flash_attention_2=True"
LOG_SUFFIX="${MODEL}_${TASKS}"

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model "qwen2_5_vl" \
    --model_args ${MODEL_ARGS}  \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${LOG_SUFFIX} \
    --output_path ./logs/
```

Check the yaml files for task names to run. The key `task` for each L2 tasks and `tag` for L1 tasks.

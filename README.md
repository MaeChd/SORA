<p align="right">
  <a href="./README_CN.md" title="简体中文"><img src="https://img.shields.io/badge/语言-简体中文-blue?logo=googletranslate" alt="简体中文"></a>
</p>

# Calibrating and Rotating: A Unified Framework for Weight Conditioning in PEFT

Authors: Da Chang, Peng Xue, Yu Li, Yongxiang Liu, Pengxiang Xu, Shixun Zhang
### News
- **[2025/11.08]**  Our article has been accepted by **AAAI2026**!
- **[2025/10.28]** The article has been uploaded to [Arxiv](https://arxiv.org/pdf/2511.00051).

### SORA
Our SORA is in `peft/tuners/sora/layer.py`.

![Framework](img/SORA_FrameWork_1.jpg)

### Usage

To reproduce, please run
```sh
bash scripts/train_and_eval.sh <adapter_name> <lr> <num_epochs> <rank> <rp> <dataset>
```
```python
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 4e-4 \
  --num_train_epochs 5 \
  --adapter_name $ADAPTER_NAME /
  --lora_rank 16 \
  --lora_alpha 32 \
  --r_p 4 \
  --output_dir /tmp/$TASK_NAME/
```

To run the evaluation of checkpoints, please run
```sh
bash scripts/eval.sh /path/to/adapter/dir <task> meta-llama/Meta-Llama-3-8B
```

To train Gemma-7B on MetaMathQA, initiate the process with the following command:

````sh
bash run_math.sh
````

**Note that the experiments for LLaMA3-8B and Gemma-7B were conducted on 8x Ascend 910C NPUs. To migrate to NVIDIA GPUs, you will need to change `npu` to `cuda`.**

To test Gemma-7B on math test benchmark, initiate the process with the following command:

```sh
bash eval_math.sh
```

**Note that you need to configure the training dataset, test dataset, base model path, and adapter path according to your local path.**


## Results
Gemma-7B was fine-tuned on MethMathQA-14k and evaluated on GSM8K, MultiArith, AQuA, SVAMP, AddSub, and SingleEq. We report the best performance across learning rates of $\{2e-4, 4e-4, 6e-6\}$. (Hardware: 8x Ascend 910C).

| Method     | Trainable Param % | GSM8K     | MultiArith | AQuA      | SVAMP    | AddSub    | SingleEq  | Avg       |
| ---------- | ------- | --------- | ---------- | --------- | -------- | --------- | --------- | --------- |
| OFT/2e-4   | 0.58    | 73.09     | **99.00**  | 38.19     | 75.00     | 86.10     | 94.69     | 77.68     |
| LoRA+/2e-4 | 0.40    | 74.83     | *98.67*    | 36.58     | 75.10     | 85.32     | 93.11     | 77.27     |
| LoRA/4e-4  | 0.40    | 74.07     | 97.50      | 37.80     | 75.90     | 85.08     | *95.85*   | 77.70     |
| DoRA/4e-4  | 0.41    | 73.77     | 97.17      | *38.34*   | **77.80** | **88.40** | 93.92     | 78.23     |
| SORA/4e-4  | 0.47    | **74.98** | 98.50      | **40.49** | *76.30*   | *87.63*   | **95.87** | **78.96** |

### Credits
* Our code is taken from https://github.com/kcc-lion/polar .

* **GLUE Benchmark** data **can be obtained** from [https://huggingface.co/datasets/nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue).
* The **Commonsense Benchmark** data **can be obtained** from [https://github.com/gkevinyen5418/LoRA-RITE](https://github.com/gkevinyen5418/LoRA-RITE).

* The `finetune_commonsense.py` script is originally adapted from https://github.com/tloen/alpaca-lora/blob/main/finetune.py.
* The lm-evaluation-harness is taken from https://github.com/EleutherAI/lm-evaluation-harness.
* The editable PEFT version is based on v0.14.1.dev0 of https://github.com/huggingface/peft.
* The `run_glue.py` script is adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py.
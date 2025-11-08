import json
import os
import sys
from functools import partial
from typing import List
import datasets
import fire
# import lm_eval
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
# from lm_eval.loggers import swanlabLogger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
import torch_npu
import swanlab

from lm_eval_prompter import generate_lm_eval_prompts
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,OFTConfig
from peft.tuners.sora.layer import SoraLinear
from prompter import Prompter


from eval_commonsense import evaluate_commonsense_datasets, WandbLogger, make_table, handle_non_serializable


def use_torch_npu():
    """
        Returns:
        True -> NPU is available, and all "cuda" related code will be 
        automatically converted "npu".
        False -> NPU is not available, and "cuda" related code will remain 
        unchanged.
    """
    try:
        import torch_npu
        npu_available = torch_npu.npu.is_available()
    except:
        npu_available = False
    # if we have torch_npu package and npu is available,
    # we can replace all "cuda" related code with "npu" by the following line
    if npu_available:
        from torch_npu.contrib import transfer_to_npu
    return npu_available


def train(
    seed: int = 0,
    # model/data params
    base_model: str = "/home/pretrained_weights/gemma-7b",  # the only required argument
    base_path: str = "/home/ft-training_set/math_10k.json",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    dtype: str = "bf16",
    # lora hyperparams
    lora_r: int = 8,
    lora_rp:int = 1, 
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    # polar hyperparams
    parameterize_S: str = "identity",
    gradient_type: str = "landing",
    regularization_lambda: float = 1.0,
    init_lora_weights: str = "default",
    # llm hyperparams
    add_eos_token: bool = True,
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # swanlab params
    swanlab_project: str = "Gemma-7B-Math-benchmark",
    swanlab_run_name: str = "",
    swanlab_watch: str = "",  # options: false | gradients | all
    swanlab_log_model: str = "",  # options: false | true
    # other
    save_strategy: str = "no",
    lr_scheduler_type: str = "cosine",
    do_eval: bool = False,
    add_stable_rank_callback: bool = False,
    activate_profiling: bool = False,
):
    
    ####################
    dataset_name = os.path.basename(data_path) if "/" in data_path else data_path
    if "math" in dataset_name:
        eval_task = "gsm8k"
        num_fewshot = 0
        log_samples = True
    else:
        # eval_task = dataset_name
        eval_task = ["boolq","piqa","social_i_qa","hellaswag","winogrande","ARC-Easy","ARC-Challenge","openbookqa"]
        num_fewshot = None
        log_samples = False

    ############################
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Fine-Tuning with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"dtype: {dtype}\n"
            f"adapter_name: {adapter_name}\n"
            f"lora_r: {lora_r}\n"
            f"regularization_lambda: {regularization_lambda}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"gradient_type: {gradient_type}\n"
            f"init_lora_weights: {init_lora_weights}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"swanlab_project: {swanlab_project}\n"
            f"swanlab_run_name: {swanlab_run_name}\n"
            f"swanlab_watch: {swanlab_watch}\n"
            f"swanlab_log_model: {swanlab_log_model}\n"
            f"eval_task: {eval_task}\n"
            f"num_fewshot: {num_fewshot}\n"
            f"init_lora_weights: {init_lora_weights}\n"
            f"save_strategy: {save_strategy}\n"
            f"lr_scheduler_type: {lr_scheduler_type}\n"
            f"do_eval: {do_eval}\n"
            f"add_stable_rank_callback: {add_stable_rank_callback}\n"
            f"activate_profiling: {activate_profiling}\n"
        )
        cli_run_args = {
            "epoch": num_epochs,
            "dataset": data_path,
            "model_name": base_model,
            "adapter": adapter_name,
            "lr": learning_rate,
            "seed": seed,
            "rank": lora_r,
            "dtype": dtype,
            "reg_lambda": regularization_lambda,
            "parametrize_S": parameterize_S,
            "gradient_type": gradient_type,
            "swanlab_name": swanlab_run_name,
            "eval_task": eval_task,
            "init_lora_weights": init_lora_weights,
            "save_strategy": save_strategy,
            "lr_scheduler_type": lr_scheduler_type,
            "do_eval": do_eval,
            "add_stable_rank_callback": add_stable_rank_callback,
            "activate_profiling": activate_profiling,
        }
        with open(os.path.join(output_dir, "cli_run_args.json"), "w") as file:
            json.dump(cli_run_args, file, indent=4)
    assert base_model, "Please specify a --base_model"
    assert dtype in ["bf16", "fp16"], f"Invalid dtype: {dtype}"
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    gradient_accumulation_steps = batch_size // micro_batch_size

    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # device_map = "auto" if world_size > 1 else {"": 0}
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": f"npu:{local_rank}"}  # 明确指定 npu
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    else:
        device_map = {"": "npu:0"}

    # Check if parameter passed or if set within environ
    use_swanlab = len(swanlab_project) > 0 or (
        "SWANLAB_PROJECT" in os.environ and len(os.environ["SWANLAB_PROJECT"]) > 0
    )
    if use_swanlab:
        swanlab_project = swanlab_project + "_" + dataset_name
        swanlab_run_name = adapter_name.lower() + f'lr-{learning_rate}-r-{lora_r}-alpha-{lora_alpha}-seed-{seed}'
    # Only overwrite environ if swanlab param passed
    if len(swanlab_project) > 0:
        os.environ["SWANLAB_PROJECT"] = swanlab_project
    if len(swanlab_watch) > 0:
        os.environ["SWANLAB_WATCH"] = swanlab_watch
    if len(swanlab_log_model) > 0:
        os.environ["SWANLAB_LOG_MODEL"] = swanlab_log_model

    transformers.set_seed(seed)
    if "llama" in base_model.lower():
        print(f"Using torch dtype: {torch_dtype}")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        gradient_checkpointing = False
        # tokenizer = LlamaTokenizer.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # if "Llama-2-7b-hf" in base_model:
        if "Meta-Llama-3-8B" in base_model:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(f"Invalid base model: {base_model}")
        model = prepare_model_for_kbit_training(model)
    elif "gemma" in base_model.lower():
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=(
                torch.bfloat16 if "gemma-3" in base_model.lower() else torch.float16
            ),
            device_map=device_map,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if "gemma-3-27b" in base_model.lower():
            gradient_checkpointing = True
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        else:
            gradient_checkpointing = False
    else:
        raise ValueError(f"Invalid base model: {base_model}")

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt
    
    if init_lora_weights == "default":
        init_strategy = True
    elif init_lora_weights == "random":
        init_strategy = False
    elif init_lora_weights == "symmetric_gaussian":
        init_strategy = "symmetric_gaussian"
    else:
        raise ValueError(f"Unknown init strategy for LoRA {init_lora_weights}")
    
    if adapter_name.lower()=='oft':
        config = OFTConfig(
             r=lora_r,
             task_type="CAUSAL_LM",
             target_modules=["q_proj", "k_proj", "v_proj"],
             block_share=True,
             module_dropout=0.0,
             init_weights=True,
        )
        # model = OFTModel(model, config,"default")
        model = get_peft_model(model,config)
    else:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            # exclude vision tower in Gemma 3
            exclude_modules=(
                r"vision_tower.vision_model.encoder.layers.\S*"
                if "gemma-3" in base_model.lower()
                else None
            ),
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            adapter_name=adapter_name,
            use_dora=True if adapter_name.lower() == "dora" else False,
            parameterize_S=parameterize_S,
            init_lora_weights=init_strategy,
        )
        if config.adapter_name.lower() == "sora":
            custom_module_mapping = {
                nn.Linear: partial(
                    SoraLinear, r_p=2,
                )
            }
            config._register_custom_module(custom_module_mapping)
        
        model = get_peft_model(model, config)

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
        test_data = None
    else:
        # train.json / test.json
        train_path = os.path.join(data_path, 'train.json')
        test_path = os.path.join(data_path, 'test.json')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Train or test file not found in {data_path}")
        
        train_load_data = load_dataset("json", data_files=train_path)
        test_load_data = load_dataset("json", data_files=test_path)

    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / total_params
    swanlab_run_name = swanlab_run_name + f"_param_{trainable_percentage:.2f}%"

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        test_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        test_data = None
        ############
        test_load_data = None
        ###########
        if test_load_data is not None:
            train_data = train_load_data['train'].shuffle().map(generate_and_tokenize_prompt)
            test_data = test_load_data['train'].map(generate_and_tokenize_prompt)
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            test_data = []
    callbacks = []
    use_bf16 = "gemma" in base_model.lower()
    if "llama" in base_model.lower():
        use_bf16 = True if dtype == "bf16" else False
    
    output_dir = output_dir + '_' +dataset_name

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        # eval_dataset=val_data,
        eval_dataset=test_data if test_data is not None else None,

        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=0.03,
            gradient_checkpointing=gradient_checkpointing,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=0.02,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 or len(test_data)> 0  else "no",
            save_strategy='steps',
            eval_steps=0.1 if val_set_size > 0 or len(test_data)> 0  else None,
            save_steps=0.1,
            ddp_find_unused_parameters=True if gradient_checkpointing else False,
            output_dir=output_dir,
            save_total_limit=2,
            seed=seed,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to="swanlab" if use_swanlab else None,
            run_name=swanlab_run_name if use_swanlab else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    if activate_profiling:
        from transformers.trainer_callback import TrainerCallback

        class ProfCallback(TrainerCallback):
            def __init__(self, prof):
                self.prof = prof

            def on_step_end(self, args, state, control, **kwargs):
                self.prof.step()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=5, wait=5, warmup=5, active=3, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(output_dir, "profiling")
            ),
            profile_memory=True,
            with_flops=False,
            with_stack=False,
            record_shapes=False,
        ) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train()
    else:
        trainer.train()

    model.save_pretrained(output_dir)

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
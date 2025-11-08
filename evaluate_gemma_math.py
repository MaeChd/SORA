import copy
import json
import os
import re
import sys
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import fire
import time
import torch
import torch_npu
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import traceback
import torch.distributed as dist

if torch_npu.npu.is_available():
    device = "npu"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass



def simple_setup():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if device == "npu":
        torch_npu.npu.set_device(local_rank)

    if world_size > 1:
        if device == "npu":
            dist.init_process_group(backend='hccl')
        elif device == "cuda":
            dist.init_process_group(backend='nccl')
        else:
            dist.init_process_group(backend='gloo')
    
    return rank, world_size, local_rank


def wait_for_all_ranks(rank, world_size):
    if world_size > 1:
        print(f'Rank {rank}: Waiting at barrier for all ranks to complete...')
        dist.barrier()
        print(f'Rank {rank}: All ranks synchronized, continuing...')
    else:
        print(f'Rank {rank}: Single process mode, no synchronization needed')


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def extract_lr_from_path(checkpoint_path: str) -> str:
    lr_pattern = r'(\d+e-\d+_\d+e-\d+)'
    match = re.search(lr_pattern, checkpoint_path)
    if match:
        return match.group(1)
    else:
        print(f"Warning: Could not extract learning rate from path: {checkpoint_path}")
        return "unknown_lr"


def extract_checkpoint_num(checkpoint_path: str) -> str:
    match = re.search(r'checkpoint-(\d+)', checkpoint_path)
    if match:
        return match.group(1)
    return "unknown_ckpt"


def extract_lora_r_from_path(checkpoint_path: str) -> str:
    pattern = r'\d+e-\d+_\d+e-\d+_(\d+)_'
    match = re.search(pattern, checkpoint_path)
    if match:
        return match.group(1)
    return "unknown_r"


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    import time
    
    args = parse_args()
    batch_size = 8
    
    rank, world_size, local_rank = simple_setup()
    is_main_process = (rank == 0)
    lr_info = extract_lr_from_path(args.lora_weights)
    ckpt_num = extract_checkpoint_num(args.lora_weights)
    lora_r_path = extract_lora_r_from_path(args.lora_weights)
    
    if is_main_process:
        print("=" * 60)
        print("Experiment Configuration:")
        print(f"  Model: {args.model}")
        print(f"  Adapter: {args.adapter}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Learning Rate: {lr_info}")
        print(f"  LoRA Rank (from path): {lora_r_path}")
        print(f"  LoRA Rank (from args): {getattr(args, 'lora_r', 'not specified')}")
        print(f"  Checkpoint: {ckpt_num}")
        print(f"  Checkpoint Path: {args.lora_weights}")
        print("=" * 60)
    

    experiment_dir = f'experiment_lr{lr_info}_r{lora_r_path}_ckpt{ckpt_num}/'
    

    save_file = f'{experiment_dir}{args.model}-{args.adapter}-{args.dataset}_rank{rank}.json'
    final_save_file = f'{experiment_dir}{args.model}-{args.adapter}-{args.dataset}.json'
    error_log_file = f'{experiment_dir}{args.model}-{args.adapter}-{args.dataset}_rank{rank}_errors.log'
    

    if is_main_process:
        create_dir(experiment_dir)
        print(f"Results will be saved to: {experiment_dir}")

    try:
        dataset = load_data(args)
        tokenizer, model = load_model(args, local_rank, world_size)
        
        # 数据分片
        dataset_per_rank = len(dataset) // world_size
        start_idx = rank * dataset_per_rank
        end_idx = start_idx + dataset_per_rank if rank < world_size - 1 else len(dataset)
        local_dataset = dataset[start_idx:end_idx]
        
        print(f"Rank {rank}: Processing samples from {start_idx} to {end_idx-1} (total: {len(local_dataset)})")
        
        total = len(local_dataset)
        correct = 0
        miss = 0.001
        output_data = []
        error_logs = []
        
        pbar = tqdm(total=total, desc=f"Rank {rank}")
        
        # 批量处理
        for batch_start in range(0, total, batch_size):
            try:
                batch_end = min(batch_start + batch_size, total)
                batch_data = local_dataset[batch_start:batch_end]
                
                instructions = [data.get('instruction') for data in batch_data]
                
                outputs, error = evaluate_batch(model, tokenizer, local_rank, instructions)
                
                if error:
                    error_log = f"Batch {batch_start}-{batch_end} failed: {error}"
                    print(f"\nRank {rank}: {error_log}")
                    error_logs.append(error_log)
                    
                    print(f"Rank {rank}: Falling back to single-sample processing...")
                    outputs = []
                    for inst in instructions:
                        single_output, single_error = evaluate_batch(model, tokenizer, local_rank, [inst])
                        if single_error:
                            outputs.append("[ERROR]")
                            error_logs.append(f"Single sample failed: {single_error}")
                        else:
                            outputs.append(single_output[0])
                
                for idx, (data, output) in enumerate(zip(batch_data, outputs)):
                    label = data.get('answer')
                    flag = False
                    
                    if output == "[ERROR]":
                        predict = None
                    elif args.dataset.lower() in ['aqua']:
                        predict = extract_answer_letter(args, output)
                        if label == predict:
                            correct += 1
                            flag = True
                    else:
                        if isinstance(label, str):
                            label = float(label)
                        predict = extract_answer_number(args, output)
                        if predict != float('inf') and abs(label - predict) <= miss:
                            correct += 1
                            flag = True
                    
                    new_data = copy.deepcopy(data)
                    new_data['output_pred'] = output
                    new_data['pred'] = predict
                    new_data['flag'] = flag
                    new_data['rank'] = rank
                    output_data.append(new_data)
                    
                    pbar.update(1)
                
                current_idx = batch_end
                if current_idx % 10 == 0 or current_idx == total:
                    print(f'\nRank {rank}: [{current_idx}/{total}] accuracy={correct/current_idx:.4f}')
                    
                    if current_idx % 40 == 0:
                        temp_result = {
                            'rank': rank,
                            'total': current_idx,
                            'correct': correct,
                            'accuracy': correct / current_idx if current_idx > 0 else 0,
                            'data': output_data,
                            'config': {
                                'learning_rate': lr_info,
                                'lora_r': lora_r_path,
                                'checkpoint': ckpt_num,
                                'adapter': args.adapter,
                                'model': args.model,
                                'dataset': args.dataset
                            }
                        }
                        with open(save_file + '.tmp', 'w') as f:
                            json.dump(temp_result, f, indent=4)
                
                if batch_start % (batch_size * 5) == 0:
                    torch_npu.npu.empty_cache()
                    
            except Exception as e:
                error_msg = f"Batch {batch_start} crashed: {str(e)}\n{traceback.format_exc()}"
                print(f"\nRank {rank}: {error_msg}")
                error_logs.append(error_msg)
                for _ in range(len(batch_data)):
                    pbar.update(1)
                continue
        
        pbar.close()
        
        rank_result = {
            'rank': rank,
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'data': output_data,
            'errors': error_logs,
            'config': {
                'learning_rate': lr_info,
                'lora_r': lora_r_path,
                'checkpoint': ckpt_num,
                'adapter': args.adapter,
                'model': args.model,
                'dataset': args.dataset,
                'checkpoint_path': args.lora_weights
            }
        }
        
        with open(save_file, 'w') as f:
            json.dump(rank_result, f, indent=4)
        
        if error_logs:
            with open(error_log_file, 'w') as f:
                f.write('\n'.join(error_logs))
        
        print(f'\nRank {rank} finished: {correct}/{total} = {correct/total:.4f}')
        print(f'Results saved to: {save_file}')
        
    except Exception as e:
        error_msg = f"Rank {rank} fatal error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open(error_log_file, 'w') as f:
            f.write(error_msg)
    
    wait_for_all_ranks(rank, world_size)
    
    if is_main_process:
        print('\n=== Merging results from all ranks ===')
        all_data = []
        total_correct = 0
        total_samples = 0
        all_errors = []
        
        for r in range(world_size):
            rank_file = f'{experiment_dir}{args.model}-{args.adapter}-{args.dataset}_rank{r}.json'
            if os.path.exists(rank_file):
                try:
                    with open(rank_file, 'r') as f:
                        rank_result = json.load(f)
                        all_data.extend(rank_result['data'])
                        total_correct += rank_result['correct']
                        total_samples += rank_result['total']
                        print(f"Rank {r}: {rank_result['correct']}/{rank_result['total']} = {rank_result['accuracy']:.4f}")
                        if 'errors' in rank_result and rank_result['errors']:
                            all_errors.extend([f"Rank {r}: {err}" for err in rank_result['errors']])
                            print(f"  Rank {r} had {len(rank_result['errors'])} errors")
                except Exception as e:
                    print(f"Error loading rank {r} results: {e}")
            else:
                print(f"WARNING: Result file for rank {r} not found: {rank_file}")
        
        final_result = {
            'total_samples': total_samples,
            'total_correct': total_correct,
            'overall_accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'world_size': world_size,
            'config': {
                'learning_rate': lr_info,
                'lora_r': lora_r_path,
                'checkpoint': ckpt_num,
                'adapter': args.adapter,
                'model': args.model,
                'dataset': args.dataset,
                'checkpoint_path': args.lora_weights
            },
            'data': all_data,
            'all_errors': all_errors
        }
        
        with open(final_save_file, 'w') as f:
            json.dump(final_result, f, indent=4)
        
        print('\n=== Final Results ===')
        print(f'Configuration:')
        print(f'  Learning Rate: {lr_info}')
        print(f'  LoRA Rank: {lora_r_path}')
        print(f'  Checkpoint: {ckpt_num}')
        print(f'Total samples: {total_samples}')
        print(f'Total correct: {total_correct}')
        print(f'Overall accuracy: {total_correct}/{total_samples} = {total_correct/total_samples:.4f}')
        print(f'Results saved to: {final_save_file}')
        
        if all_errors:
            print(f'\nTotal errors across all ranks: {len(all_errors)}')
    
    # 清理分布式环境
    cleanup_distributed()
    print(f'Rank {rank}: Cleanup complete, exiting normally')

def evaluate_batch(
        model,
        tokenizer,
        local_rank,
        instructions,
        inputs=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
):
    """批量评估,带错误处理"""
    try:
        prompts = [generate_prompt(inst, inp) for inst, inp in zip(instructions, inputs or [None]*len(instructions))]
        
        # 批量tokenize
        inputs_dict = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs_dict["input_ids"].to(f"{device}:{local_rank}")
        attention_mask = inputs_dict["attention_mask"].to(f"{device}:{local_rank}")
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        outputs = []
        for seq in generation_output.sequences:
            output = tokenizer.decode(seq, skip_special_tokens=True)
            if "### Response:" in output:
                output = output.split("### Response:")[1].strip()
            outputs.append(output)
        
        return outputs, None
    
    except Exception as e:
        error_msg = f"Error in evaluate_batch: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """


def load_data(args) -> list:
    """
    read data from dataset file
    """
    file_path = f'/home/test_datasets/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B', 'Gemma-7B', 'other'], required=True)
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--lora_r', type=int, default=None, help='LoRA rank (optional, will be extracted from path if not provided)')

    return parser.parse_args()


def load_model(args, local_rank, world_size) -> tuple:
    """
    load tuned model
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        print(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    

    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if device == "npu":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        model = model.to(f"{device}:{local_rank}")
        
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        model = model.to(f"{device}:{local_rank}")
            
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    
    return tokenizer, model


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["gsm8k", "svamp", "multiarith", "addsub", "singleeq"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    fire.Fire(main)

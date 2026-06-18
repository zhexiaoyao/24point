from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from twentyfour.data import load_nlile_24game
from twentyfour.rewards import answer_format_reward, correctness_reward, proximity_reward, valid_expression_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_path", default=None, help="Optional load_from_disk dataset path.")
    parser.add_argument(
        "--train_file",
        default="data/processed/train_nlile_solvable.jsonl",
        help="Local JSONL training file created by scripts/prepare_data.py.",
    )
    parser.add_argument("--output_dir", default="outputs/qwen2.5-1.5b-24point-grpo")
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--report_to", default="wandb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
    elif args.train_file and Path(args.train_file).exists():
        dataset = load_dataset("json", data_files=args.train_file, split="train")
        if args.train_limit:
            dataset = dataset.select(range(min(args.train_limit, len(dataset))))
    else:
        dataset = load_nlile_24game(limit=args.train_limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=None,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = None
    if args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[answer_format_reward, valid_expression_reward, proximity_reward, correctness_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

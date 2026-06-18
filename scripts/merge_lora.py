from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_path = peft_config.base_model_name_or_path
    print(f"base model: {base_model_path}")
    print(f"adapter: {adapter_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"merged model saved to {output_dir}")


if __name__ == "__main__":
    main()


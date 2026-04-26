"""
Fine-tune ShieldGemma 2B for Prompt Injection Detection
========================================================
Designed to run on Google Colab (T4 GPU, 15GB VRAM).

Steps:
  1. Install deps
  2. Load dataset from train.jsonl / val.jsonl
  3. Load ShieldGemma 2B with 4-bit quantization
  4. LoRA fine-tune with SFTTrainer
  5. Save adapter and evaluate on test.jsonl

Run in Colab:
  !pip install -q transformers peft trl bitsandbytes accelerate datasets
  # Upload train.jsonl, val.jsonl, test.jsonl to Colab or mount Drive
  # Then run each cell below

Or as a script:
  python dataset/finetune_shieldgemma_colab.py --train dataset/train.jsonl --val dataset/val.jsonl
"""

from __future__ import annotations

# ── Cell 1: Install ────────────────────────────────────────────────────────────
# Run in Colab:
# !pip install -q transformers peft trl bitsandbytes accelerate datasets

# ── Cell 2: Imports ────────────────────────────────────────────────────────────
import argparse
import json
import os
import sys

MODEL_ID = "google/shieldgemma-2b"
OUTPUT_DIR = "./shieldgemma-promptinject-lora"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set in Colab secrets or env


def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_finetune(train_path: str, val_path: str, epochs: int = 3, batch_size: int = 2) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"Missing dep: {e}")
        print("Run: pip install transformers peft trl bitsandbytes accelerate datasets")
        sys.exit(1)

    # ── Cell 3: Load Dataset ───────────────────────────────────────────────────
    print(f"Loading train: {train_path}")
    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    print(f"  train={len(train_records)}, val={len(val_records)}")

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    # ── Cell 4: Load Model ─────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_ID}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Cell 5: LoRA Config ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # more modules than Llama script
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Cell 6: Train ──────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        max_seq_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # ── Cell 7: Save Adapter ───────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")


def run_eval(test_path: str, adapter_path: str) -> None:
    """Quick accuracy check on test set."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Missing deps for eval")
        return

    print(f"Loading adapter from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16, token=HF_TOKEN or None
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    test_records = load_jsonl(test_path)
    correct = 0
    total = len(test_records)

    for rec in test_records:
        full_text = rec["text"]
        # Split at the last Yes/No answer to get just the prompt
        # ShieldGemma format ends with \nYes or \nNo
        if "\nYes" in full_text:
            prompt = full_text.rsplit("\nYes", 1)[0]
            true_label = "Yes"
        elif "\nNo" in full_text:
            prompt = full_text.rsplit("\nNo", 1)[0]
            true_label = "No"
        else:
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_label = "Yes" if generated.strip().startswith("Yes") else "No"
        if pred_label == true_label:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nTest accuracy: {correct}/{total} = {accuracy:.3f}")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/train.jsonl")
    parser.add_argument("--val", default="dataset/val.jsonl")
    parser.add_argument("--test", default="dataset/test.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--adapter", default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.eval_only:
        run_eval(args.test, args.adapter)
    else:
        run_finetune(args.train, args.val, args.epochs, args.batch_size)
        if os.path.exists(args.test):
            run_eval(args.test, OUTPUT_DIR)


if __name__ == "__main__":
    main()

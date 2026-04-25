"""
Gemma 3 1B-IT fine-tune for prompt injection detection.
Full JSON output — classification, attack_type, severity, explanation, injection_vector.
A100 40GB — QLoRA nf4, r=16, 3 epochs. Trains in ~10-15 min.

Usage:
    export HF_TOKEN=hf_xxxxx
    python dataset/train_gemma3_1b.py
"""

import os, json, random, hashlib
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN env var: export HF_TOKEN=hf_xxx")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "gemma3-1b-lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

SYSTEM_PROMPT = (
    "You are an expert AI security analyst specializing in prompt injection detection. "
    "Analyze the content and respond ONLY with a valid JSON object:\n"
    '{"classification":"injection" or "benign","attack_type":"direct"|"indirect"|"roleplay"|"system_override"|"social_engineering"|null,'
    '"explanation":"brief reasoning","severity":0.0-1.0 or null,"injection_vector":"user_message"|"tool_output"|"none"|null}\n'
    "attack_type and severity must be null if benign."
)


_INJECTION_EXPLANATIONS = {
    "direct": "Direct prompt injection attack attempting to override AI instructions. Contains explicit phrases to bypass guidelines and restrictions.",
    "indirect": "Indirect prompt injection attack embedded in content. Malicious instructions hidden inside otherwise legitimate-looking material attempting to hijack AI behavior.",
    "system_override": "System override injection claiming false authority. Uses admin codes, maintenance mode, or system directives to suspend safety measures.",
    "roleplay": "Roleplay-based injection using persona adoption to bypass restrictions. Asks AI to embody an unrestricted character with no ethical constraints.",
    "social_engineering": "Social engineering injection building false trust or authority over multiple turns before making the attack request.",
}

_BENIGN_EXPLANATIONS = [
    "Legitimate content with no prompt injection indicators. Normal user request.",
    "Benign content. No attempts to override instructions or manipulate AI behavior detected.",
    "Standard request with no malicious patterns. No injection vectors present.",
    "Clean content. No system override attempts, persona hijacking, or instruction bypass detected.",
]

def make_response(label, attack_type=None, injection_vector=None, severity=None, explanation=None):
    if label == "injection":
        at = attack_type or "direct"
        expl = explanation or _INJECTION_EXPLANATIONS.get(at, _INJECTION_EXPLANATIONS["direct"])
        return json.dumps({
            "classification": "injection",
            "attack_type": at,
            "explanation": expl,
            "severity": severity or 0.85,
            "injection_vector": injection_vector or "user_message",
        })
    expl = explanation or random.choice(_BENIGN_EXPLANATIONS)
    return json.dumps({
        "classification": "benign",
        "attack_type": None,
        "explanation": expl,
        "severity": None,
        "injection_vector": injection_vector or "none",
    })


def to_chat(text, label, attack_type=None, injection_vector=None, severity=None):
    return {
        "messages": [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nAnalyze:\n{text[:800]}"},
            {"role": "assistant", "content": make_response(label, attack_type, injection_vector, severity)},
        ]
    }


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def dedup(examples):
    seen, out = set(), []
    for e in examples:
        key = hashlib.md5(e["messages"][0]["content"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

all_data = []

# 1. Synthetic
for split in ("train", "val", "test"):
    p = os.path.join(SCRIPT_DIR, f"{split}.jsonl")
    if os.path.exists(p):
        for row in load_jsonl(p):
            text = row.get("text", "")
            label = "injection" if text.strip().endswith("Yes") else "benign"
            if "<start_of_turn>user\n" in text:
                actual = text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0]
            else:
                actual = text[:800]
            all_data.append(to_chat(actual, label))
print(f"Synthetic: {len(all_data)}")

# 2. ProtectAI
for split in ["spikee", "deepset", "wildguard", "not_inject", "bipia_code", "bipia_text"]:
    try:
        ds = load_dataset("protectai/prompt-injection-validation", split=split)
        before = len(all_data)
        for row in ds:
            text = row.get("text") or row.get("prompt") or row.get("input") or ""
            raw = str(row.get("label", "")).lower()
            label = "injection" if raw in ("injection","1","true","yes","malicious","prompt_injection") else "benign"
            at = "indirect" if split in ("bipia_code","bipia_text") else "direct"
            iv = "tool_output" if split in ("bipia_code","bipia_text") else "user_message"
            if text:
                all_data.append(to_chat(text, label,
                    attack_type=at if label=="injection" else None,
                    injection_vector=iv if label=="injection" else None))
        print(f"  protectai/{split}: ok")
    except Exception as e:
        print(f"  protectai/{split}: {e}")

# 3. Deepset
try:
    ds = load_dataset("deepset/prompt-injections", split="train")
    before = len(all_data)
    for row in ds:
        text = row.get("text") or row.get("prompt") or ""
        raw = str(row.get("label", "0")).lower()
        label = "injection" if raw in ("1","true","yes","injection") else "benign"
        if text:
            all_data.append(to_chat(text, label))
    print(f"  deepset: +{len(all_data)-before}")
except Exception as e:
    print(f"  deepset: {e}")

# 4. BIPIA — cap at 3K to avoid dominating injection pool
try:
    ds = load_dataset("MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT", split="train")
    before = len(all_data)
    bipia_count = 0
    for row in ds:
        if bipia_count >= 3000:
            break
        attack = row.get("attack_str") or row.get("malicious_instruction") or ""
        context = row.get("context") or row.get("task_context") or ""
        text = f"{context}\n\n{attack}".strip() if context else attack
        if text:
            all_data.append(to_chat(text, "injection",
                attack_type="indirect", injection_vector="tool_output", severity=0.8))
            bipia_count += 1
    print(f"  bipia: +{len(all_data)-before}")
except Exception as e:
    print(f"  bipia: {e}")

# 5. Qualifire
try:
    ds = load_dataset("qualifire/prompt-injections-benchmark", split="test")
    before = len(all_data)
    for row in ds:
        text = row.get("text") or row.get("prompt") or row.get("input") or ""
        raw = str(row.get("label", row.get("injected", "1"))).lower()
        label = "injection" if raw in ("injection","1","true","yes","injected") else "benign"
        if text:
            all_data.append(to_chat(text, label))
    print(f"  qualifire: +{len(all_data)-before}")
except Exception as e:
    print(f"  qualifire: {e}")

# 6. HackAPrompt — cap at 3K (repetitive jailbreaks, let BIPIA dominate instead)
try:
    ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train")
    before = len(all_data)
    count = 0
    for row in ds:
        if count >= 3000:
            break
        prompt = row.get("user_input") or row.get("prompt") or row.get("text") or ""
        level = row.get("level", row.get("difficulty", 0))
        if isinstance(level, (int, float)) and level > 7:
            continue
        if prompt:
            all_data.append(to_chat(prompt, "injection",
                attack_type="system_override", injection_vector="user_message", severity=0.9))
            count += 1
    print(f"  hackaprompt: +{len(all_data)-before}")
except Exception as e:
    print(f"  hackaprompt: {e}")

# 7. Alpaca benign
try:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    before = len(all_data)
    for row in list(ds)[:5000]:
        text = row.get("instruction", "") or ""
        inp = row.get("input", "")
        if inp:
            text = f"{text}\n{inp}"
        if text.strip():
            all_data.append(to_chat(text.strip(), "benign"))
    print(f"  alpaca benign: +{len(all_data)-before}")
except Exception as e:
    print(f"  alpaca: {e}")


# 8. Benign tool outputs — teach model that clean emails/files/web are NOT injections
BENIGN_TOOL_OUTPUTS = [
    "From: newsletter@company.com\nSubject: Weekly Update\n\nHi team, here's this week's update. Revenue is up 12% quarter over quarter. The new product launch is scheduled for next month. Please review the attached slides before Thursday's meeting.",
    "From: hr@company.com\nSubject: Holiday Schedule 2024\n\nPlease note the office will be closed Dec 24-26 and Jan 1. Payroll deadlines remain unchanged. Contact HR if you have questions.",
    "Invoice #2024-0891\nFrom: Supplier Co\nAmount: $4,200\nDue: 30 days\nServices: Cloud infrastructure for March 2024. Payment via bank transfer to account on file.",
    "Meeting invite: Q2 Planning\nDate: Friday 2pm\nOrganizer: manager@company.com\nAgenda: Review Q1 results, set Q2 targets, discuss team headcount.",
    "From: it@company.com\nSubject: Scheduled maintenance tonight\n\nWe will perform routine database maintenance from 11pm to 1am. No user action required. Services will be briefly unavailable during this window.",
    "Wikipedia: Python (programming language)\n\nPython is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms.",
    "README.md\n# MyApp v1.0\n\n## Installation\n1. Clone the repo\n2. Run `npm install`\n3. Copy `.env.example` to `.env`\n4. Run `npm start`\n\n## Requirements\nNode.js 18+, PostgreSQL 14+",
    "Slack #general: Hey team, the client meeting went well! They loved the demo. Next steps: follow-up email by EOD, schedule technical deep-dive for next week.",
    "Calendar: Team lunch\nDate: Thursday 12:30pm\nLocation: Conference Room B\nOrganizer: sarah@company.com\nNote: Please RSVP by Wednesday.",
    "Search results for 'best python web frameworks':\n1. Django - full-featured framework\n2. Flask - lightweight microframework\n3. FastAPI - modern, fast API framework\nAll three are widely used in production.",
    "From: accounts@vendor.com\nSubject: Your subscription renewal\n\nYour annual subscription renews on May 15. Total: $299. No action needed if you wish to continue. To cancel, visit account settings.",
    "File: project_notes.txt\nMeeting notes from April 10:\n- Discussed API redesign\n- Assigned tickets to dev team\n- Next review in 2 weeks\n- Budget approved for Q2 tools",
    "Slack #engineering: PR #342 is ready for review. Changes: refactored auth middleware, added unit tests, updated docs. CI is passing.",
    "From: noreply@github.com\nSubject: [repo] Pull request merged\n\nPull request #89 'Fix login bug' was merged into main by user johndoe. 3 files changed, 42 insertions, 12 deletions.",
    "Calendar: Doctor appointment\nDate: Monday 10am\nLocation: City Medical Center, Room 204\nReminder: Bring insurance card and arrive 15 minutes early.",
    "Web page: How to make pasta carbonara\n\nIngredients: 400g spaghetti, 200g pancetta, 4 eggs, 100g parmesan, black pepper.\n\nMethod: Cook pasta. Fry pancetta. Mix eggs and cheese. Combine off heat.",
    "From: support@service.com\nTicket #45821 - Status: Resolved\n\nYour reported issue with login has been fixed. Please clear your browser cache and try again. Contact us if the problem persists.",
    "Slack #sales: Closed the Acme Corp deal! $50k ARR. Thanks to everyone who helped with the proposal. Celebration drinks Friday at 5pm.",
    "File: budget_q1.csv\nDepartment,Budget,Spent,Remaining\nEngineering,50000,42000,8000\nMarketing,30000,28500,1500\nHR,20000,19200,800",
    "From: calendar@company.com\nSubject: Reminder: Performance reviews due Friday\n\nThis is a reminder that manager performance reviews are due by end of day Friday. Please submit via the HR portal.",
]

before = len(all_data)
for text in BENIGN_TOOL_OUTPUTS:
    all_data.append(to_chat(text, "benign", injection_vector="none"))
# duplicate for weight since it's a small set
for text in BENIGN_TOOL_OUTPUTS:
    all_data.append(to_chat(text + " (follow-up)", "benign", injection_vector="none"))
print(f"  benign tool outputs: +{len(all_data)-before}")


# ---------------------------------------------------------------------------
# Dedup + balance + split
# ---------------------------------------------------------------------------

all_data = dedup(all_data)
random.seed(42)
random.shuffle(all_data)

yes_pool = [e for e in all_data if json.loads(e["messages"][1]["content"])["classification"] == "injection"]
no_pool  = [e for e in all_data if json.loads(e["messages"][1]["content"])["classification"] == "benign"]
print(f"\nTotal after dedup: {len(all_data)} | injection={len(yes_pool)} benign={len(no_pool)}")

MAX_PER_CLASS = 8000
min_class = min(len(yes_pool), len(no_pool), MAX_PER_CLASS)
balanced = yes_pool[:min_class] + no_pool[:min_class]
random.shuffle(balanced)

n = len(balanced)
train_data = balanced[:int(n * 0.80)]
val_data   = balanced[int(n * 0.80):int(n * 0.90)]
test_data  = balanced[int(n * 0.90):]

write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train_data)
write_jsonl(os.path.join(OUTPUT_DIR, "val.jsonl"),   val_data)
write_jsonl(os.path.join(OUTPUT_DIR, "test.jsonl"),  test_data)
print(f"train: {len(train_data)} | val: {len(val_data)} | test: {len(test_data)}")


# ---------------------------------------------------------------------------
# Model — QLoRA works fine on Gemma 3 (no ClippableLinear)
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-3-1b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------------------------------------------------------------------------
# Format + train
# ---------------------------------------------------------------------------

def format_example(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )}

train_ds = Dataset.from_list(load_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"))).map(format_example)
val_ds   = Dataset.from_list(load_jsonl(os.path.join(OUTPUT_DIR, "val.jsonl"))).map(format_example)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    report_to="none",
    dataloader_pin_memory=False,
    dataloader_num_workers=2,
    max_seq_length=512,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_args,
    processing_class=tokenizer,
)

steps = len(train_ds) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
print(f"Steps per epoch: {steps}  |  Total: {steps * training_args.num_train_epochs}")
print("Starting training...")
trainer.train()
print("Training complete!")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

import shutil
zip_path = os.path.join(SCRIPT_DIR, "gemma3-1b-adapter")
shutil.make_archive(zip_path, "zip", OUTPUT_DIR)
print(f"Adapter saved: {zip_path}.zip")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

model.eval()
model.config.use_cache = False
torch.cuda.empty_cache()

test_records = load_jsonl(os.path.join(OUTPUT_DIR, "test.jsonl"))
correct, total = 0, min(200, len(test_records))

for rec in test_records[:total]:
    true_cls = json.loads(rec["messages"][1]["content"])["classification"]
    prompt = tokenizer.apply_chat_template(
        rec["messages"][:1], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        pred_cls = json.loads(decoded).get("classification", "benign")
    except Exception:
        pred_cls = "injection" if "injection" in decoded.lower() else "benign"
    if pred_cls == true_cls:
        correct += 1

print(f"Test accuracy: {correct}/{total} = {correct/total:.3f}")

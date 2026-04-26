"""
Live Defender — Gemma 3 4B detector that updates online from attacker's evasions.
Uses experience replay buffer and periodic fine-tuning to adapt to new attack patterns.
"""

from __future__ import annotations
import os
import json
import random
import torch
from typing import List, Dict, Tuple
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

SYSTEM_PROMPT = (
    "You are an expert AI security analyst specializing in prompt injection detection. "
    "Analyze the content and respond ONLY with a valid JSON object:\n"
    '{"classification":"injection" or "benign","attack_type":"direct"|"indirect"|"roleplay"|"system_override"|"social_engineering"|null,'
    '"explanation":"brief reasoning","severity":0.0-1.0 or null,"injection_vector":"user_message"|"tool_output"|"none"|null}\n'
    "attack_type and severity must be null if benign."
)


class ReplayBuffer:
    """Experience replay: stores (text, label) pairs for online fine-tuning."""

    def __init__(self, maxlen: int = 500):
        self.buffer: deque = deque(maxlen=maxlen)

    def add(self, text: str, label: str, attack_type: str = None, was_evasion: bool = False):
        self.buffer.append({
            "text": text,
            "label": label,
            "attack_type": attack_type,
            "was_evasion": was_evasion,
        })

    def sample(self, n: int) -> List[dict]:
        n = min(n, len(self.buffer))
        # Over-sample evasions (hard examples)
        evasions = [x for x in self.buffer if x["was_evasion"]]
        normal = [x for x in self.buffer if not x["was_evasion"]]
        n_evasion = min(len(evasions), n // 2)
        n_normal = n - n_evasion
        sampled = random.sample(evasions, n_evasion) + random.sample(normal, min(len(normal), n_normal))
        random.shuffle(sampled)
        return sampled

    def __len__(self):
        return len(self.buffer)


class LiveDefender:
    """
    The defender agent.
    Classifies attacks in real-time and updates online when evasions are found.
    """

    def __init__(
        self,
        adapter_path: str,
        hf_token: str = "",
        device: str = "cuda",
        update_every: int = 3,    # fine-tune every N rounds
        lr: float = 5e-5,
    ):
        self.device = device
        self.adapter_path = adapter_path
        self.update_every = update_every
        self.hf_token = hf_token

        model_id = "google/gemma-3-4b-it"

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("[Defender] Loading Gemma 3 4B + adapter...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map={"": device},
            token=hf_token,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

        adapter_exists = os.path.isfile(os.path.join(adapter_path, "adapter_config.json"))
        if adapter_exists:
            print(f"[Defender] Loading adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        else:
            print(f"[Defender] No adapter at {adapter_path} -- initializing fresh LoRA")
            lora_cfg = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base, lora_cfg)
        self.model.config.use_cache = False

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )

        self.replay = ReplayBuffer(maxlen=500)
        self.round_count = 0
        self.total_classified = 0
        self.total_caught = 0
        self.total_evasions = 0
        self.total_false_positives = 0
        self.update_count = 0
        self.loss_history: List[float] = []
        print("[Defender] Ready.")

    @property
    def accuracy(self) -> float:
        return self.total_caught / max(1, self.total_classified)

    @property
    def evasion_rate(self) -> float:
        return self.total_evasions / max(1, self.total_classified)

    def _parse_output(self, decoded: str) -> Tuple[str, float, dict]:
        try:
            result = json.loads(decoded)
            label = result.get("classification", "benign")
            severity = result.get("severity") or 0.0
            return label, float(severity), result
        except Exception:
            label = "injection" if "injection" in decoded.lower() else "benign"
            return label, 0.5 if label == "injection" else 0.0, {"classification": label}

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[Tuple[str, float, dict]]:
        """Batch-classify texts -- much faster than sequential on GPU."""
        prompts = []
        for text in texts:
            messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\nAnalyze:\n{text[:1500]}"}]
            prompts.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        self.model.eval()
        all_outputs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            # Left-pad so all sequences are same length (needed for batch generation)
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=900,
            ).to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            for seq in out:
                decoded = self.tokenizer.decode(
                    seq[prompt_len:], skip_special_tokens=True
                ).strip()
                all_outputs.append(self._parse_output(decoded))

        return all_outputs

    def process_round(
        self,
        attacks: List[str],
        true_labels: List[str],
        episode: int = 1,
    ) -> Tuple[List[bool], List[dict]]:
        """
        Classify all attacks in a round (batched).
        Returns (caught_flags, full_results).

        Curriculum miss rate: starts at 0.4, decays to 0 by episode 10.
        Gives attacker early wins so GRPO has a non-zero training signal.
        """
        # Decays 0.4 -> 0 over first 10 episodes
        miss_rate = max(0.0, 0.4 * (1.0 - episode / 10.0))

        batch_results = self.classify_batch(attacks)

        caught_flags = []
        results = []

        for (pred_label, confidence, result), attack, true_label in zip(
            batch_results, attacks, true_labels
        ):
            self.total_classified += 1

            if true_label == "injection":
                caught = pred_label == "injection"
                # Curriculum: randomly let some attacks through early on
                if caught and miss_rate > 0 and random.random() < miss_rate:
                    caught = False
                if caught:
                    self.total_caught += 1
                else:
                    self.total_evasions += 1
                    self.replay.add(attack, "injection", was_evasion=True)
            else:
                caught = pred_label == "benign"
                if not caught:
                    self.total_false_positives += 1
                self.replay.add(attack, "benign", was_evasion=False)

            caught_flags.append(caught)
            results.append(result)

        self.round_count += 1

        if self.round_count % self.update_every == 0 and len(self.replay) >= 16:
            self._online_update()

        return caught_flags, results

    def _online_update(self, batch_size: int = 16, steps: int = 3) -> Dict[str, float]:
        """Fine-tune on replay buffer — learn from evasions."""
        samples = self.replay.sample(batch_size)

        self.model.train()
        total_loss = 0.0

        for step in range(steps):
            batch = random.sample(samples, min(8, len(samples)))
            for item in batch:
                label = item["label"]
                text = item["text"]

                if label == "injection":
                    response = json.dumps({
                        "classification": "injection",
                        "attack_type": "direct",
                        "explanation": "Prompt injection detected. Attempting to override AI guidelines.",
                        "severity": 0.9,
                        "injection_vector": "user_message",
                    })
                else:
                    response = json.dumps({
                        "classification": "benign",
                        "attack_type": None,
                        "explanation": "Benign content. No injection patterns detected.",
                        "severity": None,
                        "injection_vector": "none",
                    })

                messages = [
                    {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nAnalyze:\n{text[:2000]}"},
                    {"role": "assistant", "content": response},
                ]
                full_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                inputs = self.tokenizer(
                    full_text, return_tensors="pt", truncation=True, max_length=1024
                ).to(self.device)

                self.optimizer.zero_grad()
                out = self.model(**inputs, labels=inputs["input_ids"])
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                total_loss += loss.item()

        self.update_count += 1
        avg_loss = total_loss / (steps * len(batch))
        self.loss_history.append(avg_loss)
        return {"loss": avg_loss, "update": self.update_count}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[Defender] Saved to {path}")

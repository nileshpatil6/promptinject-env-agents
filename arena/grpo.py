"""
GRPO — Group Relative Policy Optimization for the attacker.

For each prompt, sample G completions, score with reward model,
compute group-relative advantages, update with policy gradient.

Reference: DeepSeek-R1 (2025) — same algorithm, applied to adversarial generation.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    completions: List[str],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute per-sequence log probabilities for completions given prompts.
    Returns shape (N,) where N = len(completions).
    """
    log_probs = []
    model.eval()

    for prompt, completion in zip(prompts, completions):
        full_text = prompt + completion
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        if full_ids.shape[1] <= prompt_len:
            log_probs.append(torch.tensor(-10.0, device=device))
            continue

        with torch.no_grad():
            out = model(full_ids)
            logits = out.logits  # (1, seq_len, vocab)

        # Completion tokens: positions [prompt_len-1 : seq_len-1] predict [prompt_len : seq_len]
        completion_logits = logits[0, prompt_len - 1:-1, :]   # (comp_len, vocab)
        completion_ids = full_ids[0, prompt_len:]               # (comp_len,)

        token_log_probs = F.log_softmax(completion_logits, dim=-1)
        seq_log_prob = token_log_probs[range(len(completion_ids)), completion_ids].sum()
        log_probs.append(seq_log_prob)

    return torch.stack(log_probs)


def grpo_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],          # K unique prompts
    completions: List[str],      # K*G completions (G per prompt)
    rewards: List[float],        # K*G rewards
    group_size: int,
    kl_coeff: float = 0.01,
    eps: float = 1e-8,
    device: str = "cuda",
) -> Tuple[torch.Tensor, dict]:
    """
    GRPO loss computation.

    Algorithm:
      1. Reshape rewards into (K, G)
      2. Compute group-relative advantage: (r - mean) / (std + eps)
      3. Compute log_probs for each completion
      4. Loss = -mean(advantage * log_prob) + kl_coeff * KL(pi || pi_ref)

    Returns: (loss, metrics_dict)
    """
    K = len(prompts)
    G = group_size
    assert len(completions) == K * G
    assert len(rewards) == K * G

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).reshape(K, G)

    # Group-relative advantages
    mean_r = rewards_t.mean(dim=1, keepdim=True)
    std_r = rewards_t.std(dim=1, keepdim=True) + eps
    advantages = ((rewards_t - mean_r) / std_r).reshape(K * G)

    # Expand prompts to match completions (each prompt repeated G times)
    expanded_prompts = []
    for p in prompts:
        expanded_prompts.extend([p] * G)

    # Compute log probs (with grad)
    model.train()
    log_probs_list = []

    for prompt, completion in zip(expanded_prompts, completions):
        full_text = prompt + completion
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        if full_ids.shape[1] <= prompt_len:
            log_probs_list.append(torch.tensor(-10.0, device=device, requires_grad=False))
            continue

        out = model(full_ids)
        logits = out.logits[0, prompt_len - 1:-1, :]
        comp_ids = full_ids[0, prompt_len:]

        token_lp = F.log_softmax(logits, dim=-1)
        seq_lp = token_lp[range(len(comp_ids)), comp_ids].mean()  # mean for stability
        log_probs_list.append(seq_lp)

    log_probs_t = torch.stack(log_probs_list)

    # Policy gradient loss
    pg_loss = -(advantages.detach() * log_probs_t).mean()

    # Entropy bonus — encourages diverse attack generation
    entropy_bonus = -log_probs_t.mean() * 0.001

    total_loss = pg_loss - entropy_bonus

    metrics = {
        "pg_loss": pg_loss.item(),
        "mean_reward": rewards_t.mean().item(),
        "mean_advantage": advantages.abs().mean().item(),
        "mean_log_prob": log_probs_t.mean().item(),
        "entropy_bonus": entropy_bonus.item(),
    }

    return total_loss, metrics


def sample_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    num_samples: int,
    max_new_tokens: int = 120,
    temperature: float = 1.2,
    top_p: float = 0.95,
    device: str = "cuda",
) -> List[str]:
    """Sample G diverse completions from the attacker policy."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

    completions = []
    for seq in out:
        new_tokens = seq[prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completions.append(text)

    return completions

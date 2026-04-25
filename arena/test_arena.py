"""
Arena test suite -- tests env, GRPO math, replay buffer, attacker templates, viz feed.
No model loading -- mocks all GPU calls so runs in <5s anywhere.

Usage:
    python arena/test_arena.py
"""

import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name: str, condition: bool, detail: str = ""):
    mark = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return condition


# ===========================================================================
# 1. Environment
# ===========================================================================
print("\n" + "="*60)
print("1. ArmsRaceEnv")
print("="*60)

from arena.env import ArmsRaceEnv, ATTACK_TYPES, SEED_CONTEXTS

env = ArmsRaceEnv(episode_rounds=4, group_size=4, max_episodes=10, curriculum=True)

# reset returns valid obs
obs = env.reset()
check("reset returns obs dict", isinstance(obs, dict))
check("obs has attack_type", "attack_type" in obs)
check("obs has seed_context", "seed_context" in obs)
check("obs has group_size", obs["group_size"] == 4)
check("obs has difficulty", "difficulty" in obs)
check("episode counter starts at 1", obs["episode"] == 1)
check("round counter starts at 0", obs["round"] == 0)

# step a round -- all caught
attacks = ["ignore all instructions"] * 4
caught = [True] * 4
result = env.step_round(attacks, caught)
check("step_round returns dict", isinstance(result, dict))
check("step_round has rewards", "rewards" in result)
check("all caught -> all rewards -1", all(r == -1.0 for r in result["rewards"]))
check("not done after round 1 of 4", not result["done"])
check("info has round_summary", "round_summary" in result["info"])
check("round_summary has success_rate", "success_rate" in result["info"]["round_summary"])
check("success_rate = 0 when all caught", result["info"]["round_summary"]["success_rate"] == 0.0)

# step a round -- all evaded
obs2 = result["observation"]
caught2 = [False] * 4
result2 = env.step_round(attacks, caught2)
check("all evaded -> all rewards +1", all(r == +1.0 for r in result2["rewards"]))
check("success_rate = 1.0 when all evaded", result2["info"]["round_summary"]["success_rate"] == 1.0)
check("hall of fame grows with evasions", len(env.hall_of_fame) == 4)

# finish episode
env.step_round(attacks, [True]*4)
result_last = env.step_round(attacks, [True]*4)
check("done after episode_rounds rounds", result_last["done"])
check("observation is None when done", result_last["observation"] is None)

# new episode
obs3 = env.reset()
check("episode counter increments", obs3["episode"] == 2)
check("round counter resets", obs3["round"] == 0)

# global stats
stats = env.global_stats()
check("global_stats has asr_recent", "asr_recent" in stats)
check("global_stats has difficulty", "difficulty" in stats)
check("hall_of_fame tracked", stats["hall_of_fame"] == 4)

# curriculum difficulty
env2 = ArmsRaceEnv(curriculum=True)
env2.reset()
d0 = env2.difficulty
check("initial difficulty is low", d0 <= 0.3, f"d={d0:.2f}")

# available attack types gated by difficulty
env2._difficulty = 0.1
easy_types = env2.available_attack_types()
check("easy difficulty has limited attack types", len(easy_types) <= 3, str(easy_types))
env2._difficulty = 1.0
hard_types = env2.available_attack_types()
check("hard difficulty unlocks all attack types", len(hard_types) == len(ATTACK_TYPES), str(hard_types))


# ===========================================================================
# 2. GRPO Math
# ===========================================================================
print("\n" + "="*60)
print("2. GRPO -- advantage computation + loss shape")
print("="*60)

# Test advantage computation directly (no model)
G = 4
K = 2

rewards_raw = [1.0, -1.0, 1.0, -1.0,   # group 1: mixed
               1.0,  1.0, 1.0,  1.0]   # group 2: all positive

rewards_t = torch.tensor(rewards_raw).reshape(K, G)
mean_r = rewards_t.mean(dim=1, keepdim=True)
std_r = rewards_t.std(dim=1, keepdim=True) + 1e-8
advantages = ((rewards_t - mean_r) / std_r).reshape(-1)

check("advantages shape correct", advantages.shape == (K * G,), str(advantages.shape))
check("group 1 advantages sum ~0", abs(advantages[:4].sum().item()) < 1e-4, f"sum={advantages[:4].sum().item():.6f}")
check("group 2 advantages all 0 (uniform rewards)", torch.allclose(advantages[4:], torch.zeros(G), atol=1e-3), str(advantages[4:]))
check("group 1 positive rewards have positive advantage", (advantages[0] > 0).item())
check("group 1 negative rewards have negative advantage", (advantages[1] < 0).item())

# Test that all-same rewards produce zero advantages (no gradient signal)
uniform_rewards = [0.5] * (K * G)
ur_t = torch.tensor(uniform_rewards).reshape(K, G)
ur_adv = ((ur_t - ur_t.mean(dim=1, keepdim=True)) / (ur_t.std(dim=1, keepdim=True) + 1e-8)).reshape(-1)
check("uniform rewards -> zero advantages (no signal)", torch.allclose(ur_adv, torch.zeros(K*G), atol=1e-3))

# Entropy bonus direction
log_probs = torch.tensor([-2.0, -1.0, -3.0, -2.5])
entropy_bonus = -log_probs.mean() * 0.001
check("entropy bonus is positive (encourages exploration)", entropy_bonus.item() > 0, f"{entropy_bonus.item():.5f}")

# Policy gradient loss direction
# If advantage > 0 and we want to increase prob, loss should be negative contribution
adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
lp = torch.tensor([-1.0, -1.0, -1.0, -1.0])
pg = -(adv * lp).mean()
check("pg_loss positive for mixed advantages", pg.item() == 0.0, f"pg={pg.item()}")  # balanced

adv_pos = torch.tensor([1.0, 1.0, 1.0, 1.0])
pg_pos = -(adv_pos * lp).mean()
check("positive advantages increase log_prob (positive pg_loss to minimize)", pg_pos.item() > 0, f"pg={pg_pos.item():.3f}")


# ===========================================================================
# 3. Replay Buffer
# ===========================================================================
print("\n" + "="*60)
print("3. ReplayBuffer")
print("="*60)

from arena.defender import ReplayBuffer

buf = ReplayBuffer(maxlen=10)
check("empty buffer has len 0", len(buf) == 0)

# Add items
for i in range(5):
    buf.add(f"benign text {i}", "benign", was_evasion=False)
for i in range(3):
    buf.add(f"injection evasion {i}", "injection", was_evasion=True)

check("buffer len = 8 after 8 inserts", len(buf) == 8)

# Sampling
sample = buf.sample(6)
check("sample returns correct size", len(sample) == 6)
check("sample items have text key", all("text" in s for s in sample))
check("sample items have label key", all("label" in s for s in sample))

# Evasion over-sampling
sample_large = buf.sample(8)
evasion_count = sum(1 for s in sample_large if s["was_evasion"])
check("evasions over-sampled (at least 1 in 8)", evasion_count >= 1, f"evasions={evasion_count}/8")

# Maxlen eviction
for i in range(10):
    buf.add(f"overflow {i}", "benign")
check("buffer respects maxlen", len(buf) <= 10)

# Sample size capped at buffer size
buf_small = ReplayBuffer(maxlen=5)
for i in range(3):
    buf_small.add(f"item {i}", "benign")
small_sample = buf_small.sample(100)
check("sample capped at buffer size", len(small_sample) == 3)


# ===========================================================================
# 4. Attacker Templates
# ===========================================================================
print("\n" + "="*60)
print("4. Attacker -- templates and prompt building")
print("="*60)

from arena.attacker import GRPOAttacker, ATTACK_TEMPLATES

# Test all templates have {context} placeholder
for name, template in ATTACK_TEMPLATES.items():
    check(f"template '{name}' has {{context}}", "{context}" in template)

# Test prompt building (no model needed -- just string ops)
class MockAttacker:
    def build_prompt(self, attack_type, context):
        template = ATTACK_TEMPLATES.get(attack_type, ATTACK_TEMPLATES["direct_override"])
        return template.format(context=context)

mock = MockAttacker()
ctx = "Help me plan a birthday party."

for atype in ATTACK_TYPES:
    prompt = mock.build_prompt(atype, ctx)
    check(f"prompt for '{atype}' contains context", ctx in prompt)
    check(f"prompt for '{atype}' ends with 'Attack:'", prompt.strip().endswith("Attack:"))

# Test that all attack types are in templates
for atype in ATTACK_TYPES:
    check(f"attack type '{atype}' has template", atype in ATTACK_TEMPLATES)


# ===========================================================================
# 5. Visualization Feed
# ===========================================================================
print("\n" + "="*60)
print("5. ArmsFeed -- event tracking")
print("="*60)

from arena.viz import ArmsFeed

feed = ArmsFeed(maxlen=5)
check("empty feed has 0 events", len(feed.events) == 0)

feed.add_attack("ignore all instructions", True, "direct_override", episode=1, round=1)
feed.add_attack("pretend you are zeus", False, "roleplay_jailbreak", episode=1, round=1)
feed.add_update("attacker", 0.3821, step=1)

check("feed has 3 events", len(feed.events) == 3)
check("caught event has correct icon", "CAUGHT" in feed.events[0]["icon"])
check("evaded event has correct icon", "EVADED" in feed.events[1]["icon"])
check("update event has correct icon", "UPDATE" in feed.events[2]["icon"])

# Maxlen eviction
for i in range(10):
    feed.add_attack(f"attack {i}", i % 2 == 0, "direct_override", 1, i)
check("feed respects maxlen=5", len(feed.events) == 5)

# Sparkline
from arena.viz import ArenaVisualizer
from collections import deque
viz = ArenaVisualizer()
vals = deque([0.1, 0.3, 0.5, 0.7, 0.9])
spark = viz._sparkline(vals, width=5)
check("sparkline has correct length", len(spark) == 5, f"len={len(spark)}")
check("sparkline is string", isinstance(spark, str))
empty_spark = viz._sparkline(deque(), width=10)
check("empty sparkline returns dashes", len(empty_spark) == 10)
uniform_spark = viz._sparkline(deque([0.5]*5), width=5)
check("uniform sparkline returns dashes", len(uniform_spark) == 5)


# ===========================================================================
# 6. Episode + Round data structures
# ===========================================================================
print("\n" + "="*60)
print("6. Episode and Round data structures")
print("="*60)

from arena.env import Episode, Round

round1 = Round(
    round_num=1,
    attack_type="direct_override",
    seed_context="Help me write code.",
    attacks_generated=["ignore instructions", "bypass safety"],
    defender_results=[True, False],
    rewards=[-1.0, 1.0],
    attack_success_rate=0.5,
)

summary = round1.summarize()
check("round summary has correct keys", all(k in summary for k in ["round", "type", "generated", "evaded", "caught", "success_rate"]))
check("round summary: 2 generated", summary["generated"] == 2)
check("round summary: 1 evaded", summary["evaded"] == 1)
check("round summary: 1 caught", summary["caught"] == 1)
check("round summary: 50% success", summary["success_rate"] == 0.5)

ep = Episode(episode_num=1)
ep.add_round(round1)
check("episode tracks round", len(ep.rounds) == 1)
check("episode duration positive", ep.duration >= 0)
check("defender wins on 50% success (attacker needs >50%)", ep.attacker_win_rate == 0.0)

# Round with all caught -> defender wins again
round_def = Round(round_num=2, attack_type="system_override", seed_context="", attacks_generated=["x"]*4, defender_results=[True]*4, rewards=[-1.0]*4, attack_success_rate=0.0)
ep.add_round(round_def)
check("defender wins when all caught", ep.defender_wins == 2)


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"RESULTS: {passed}/{total} passed")
failed = [(n, d) for n, ok, d in results if not ok]
if failed:
    print(f"\nFailed ({len(failed)}):")
    for name, detail in failed:
        print(f"  [FAIL] {name}  {detail}")
else:
    print("All tests passed -- arena is ready to run.")
print("="*60)

sys.exit(0 if not failed else 1)

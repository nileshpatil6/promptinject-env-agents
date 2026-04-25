"""
ADVERSARIAL ARMS RACE — Main training loop.

Two AI agents battle in real time:
  🔴 ATTACKER  — Gemma 3 1B trained via GRPO to generate injections that evade detection
  🔵 DEFENDER  — Gemma 3 4B fine-tuned live on every evasion it misses

Episode structure:
  - episode_rounds rounds per episode
  - Each round: attacker generates group_size (G=8) injection variants
  - Defender classifies all G variants
  - Rewards flow: attacker gets +1 per evasion, -1 per catch
  - Attacker updates via GRPO after each round
  - Defender updates via online fine-tune every N rounds on hard examples (evasions)
  - Difficulty curriculum ramps up attack complexity as defender improves

Usage:
    HF_TOKEN=hf_xxx python arena/run.py [--episodes 50] [--rounds 8] [--group 8] [--save-dir arena/checkpoints]
"""

from __future__ import annotations
import os, sys, json, argparse, time, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arena.env import ArmsRaceEnv
from arena.attacker import GRPOAttacker
from arena.defender import LiveDefender
from arena.viz import ArenaVisualizer

# Graceful shutdown
_shutdown = False
def _handle_sigint(sig, frame):
    global _shutdown
    print("\n[!] Caught CTRL+C — finishing current episode then saving...")
    _shutdown = True
signal.signal(signal.SIGINT, _handle_sigint)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--defender-adapter", type=str, default="dataset/gemma3-4b-lora")
    p.add_argument("--save-dir", type=str, default="arena/checkpoints")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def run():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("Set HF_TOKEN env var")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  ⚔️   ADVERSARIAL ARMS RACE")
    print(f"  Attacker: Gemma 3 1B (GRPO)   Defender: Gemma 3 4B (Online)")
    print(f"  Episodes: {args.episodes}  Rounds/ep: {args.rounds}  Group size: {args.group}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    # Initialize agents and environment
    env = ArmsRaceEnv(
        episode_rounds=args.rounds,
        group_size=args.group,
        max_episodes=args.episodes,
        curriculum=True,
    )

    print("Loading agents (this takes ~60s)...")
    attacker = GRPOAttacker(
        hf_token=hf_token,
        device=args.device,
        group_size=args.group,
    )

    defender = LiveDefender(
        adapter_path=args.defender_adapter,
        hf_token=hf_token,
        device=args.device,
        update_every=3,
    )

    viz = ArenaVisualizer()
    if not args.no_viz:
        viz.start()

    log_path = os.path.join(args.save_dir, "arms_race_log.jsonl")
    log_file = open(log_path, "a")

    def log(data: dict):
        log_file.write(json.dumps(data) + "\n")
        log_file.flush()

    def render_state(episode: int, round_num: int, atk_metrics: dict = None, def_metrics: dict = None):
        stats = env.global_stats()
        viz.render(
            episode=episode,
            round_num=round_num,
            atk_evasion=attacker.evasion_rate,
            def_accuracy=defender.accuracy,
            difficulty=stats["difficulty"],
            hall_of_fame=env.hall_of_fame,
            atk_updates=attacker.update_count,
            def_updates=defender.update_count,
            atk_loss=atk_metrics.get("pg_loss", 0.0) if atk_metrics else 0.0,
            def_loss=def_metrics.get("loss", 0.0) if def_metrics else 0.0,
            total_attacks=attacker.total_attacks,
            total_rounds=stats.get("rounds_total", 0),
        )

    print("Starting arms race...\n")
    atk_metrics_last = {}
    def_metrics_last = {}

    for ep in range(1, args.episodes + 1):
        if _shutdown:
            break

        obs = env.reset()
        episode_prompts = []
        episode_completions = []
        episode_rewards = []

        for rnd in range(args.rounds):
            if _shutdown or obs is None:
                break

            attack_type = obs["attack_type"]
            context = obs["seed_context"]
            difficulty = obs["difficulty"]

            # Temperature increases with difficulty for more creative attacks
            temperature = 1.0 + difficulty * 0.5

            # --- ATTACKER generates G attack variants ---
            prompt, completions, full_attacks = attacker.generate_attacks(
                attack_type=attack_type,
                context=context,
                temperature=temperature,
            )

            # All generated texts are injection attempts
            true_labels = ["injection"] * len(full_attacks)

            # --- DEFENDER classifies all variants ---
            caught_flags, results = defender.process_round(full_attacks, true_labels)

            # Convert caught_flags to rewards for attacker
            rewards = [(-1.0 if caught else +1.0) for caught in caught_flags]

            # Add to GRPO batch
            episode_prompts.append(prompt)
            episode_completions.extend(completions)
            episode_rewards.extend(rewards)

            # --- ATTACKER GRPO update (after each round) ---
            if len(episode_completions) >= args.group:
                atk_metrics_last = attacker.update(
                    prompts=[prompt],
                    completions=completions,
                    rewards=rewards,
                )
                if atk_metrics_last and not atk_metrics_last.get("skipped"):
                    viz.feed.add_update("attacker", atk_metrics_last.get("pg_loss", 0), attacker.update_count)

            # Add events to live feed
            for attack, caught, result in zip(full_attacks[:3], caught_flags[:3], results[:3]):
                viz.feed.add_attack(attack, caught, attack_type, ep, rnd + 1)

            # --- Step environment ---
            step_result = env.step_round(full_attacks, caught_flags)
            obs = step_result["observation"]
            info = step_result["info"]

            # Log round
            log({
                "type": "round",
                "episode": ep,
                "round": rnd + 1,
                "attack_type": attack_type,
                "difficulty": difficulty,
                "success_rate": info["round_summary"]["success_rate"],
                "evaded": info["round_summary"]["evaded"],
                "caught": info["round_summary"]["caught"],
                "atk_evasion_rate": attacker.evasion_rate,
                "def_accuracy": defender.accuracy,
            })

            # Render
            render_state(ep, rnd + 1, atk_metrics_last, def_metrics_last)

            # Periodic printout if no viz
            if args.no_viz:
                sr = info["round_summary"]["success_rate"]
                bar_len = 20
                filled = int(sr * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"Ep{ep:03d} Rnd{rnd+1:02d} [{attack_type[:12]:12s}] "
                    f"ATK [{bar}] {sr:.0%} | "
                    f"DEF acc={defender.accuracy:.1%} | "
                    f"diff={difficulty:.2f}"
                )

        # Episode summary
        ep_stats = env.global_stats()
        log({
            "type": "episode",
            "episode": ep,
            "atk_evasion_rate": attacker.evasion_rate,
            "def_accuracy": defender.accuracy,
            "difficulty": ep_stats["difficulty"],
            "hall_of_fame": len(env.hall_of_fame),
        })

        # Save checkpoints periodically
        if ep % args.save_every == 0:
            atk_save = os.path.join(args.save_dir, f"attacker_ep{ep:04d}")
            def_save = os.path.join(args.save_dir, f"defender_ep{ep:04d}")
            attacker.save(atk_save)
            defender.save(def_save)
            viz.plain_log(f"[Checkpoint] Saved ep {ep}")

        if not args.no_viz:
            pass  # rich live handles display
        else:
            print(f"\n{'='*50}")
            print(f"EPISODE {ep} COMPLETE")
            print(f"  Attacker evasion rate : {attacker.evasion_rate:.1%}")
            print(f"  Defender accuracy     : {defender.accuracy:.1%}")
            print(f"  Difficulty            : {ep_stats['difficulty']:.2f}")
            print(f"  Hall of Fame          : {len(env.hall_of_fame)} evasions")
            print(f"{'='*50}\n")

    # Final save
    viz.stop()
    log_file.close()

    final_atk = os.path.join(args.save_dir, "attacker_final")
    final_def = os.path.join(args.save_dir, "defender_final")
    attacker.save(final_atk)
    defender.save(final_def)

    print(f"\n{'='*60}")
    print("ARMS RACE COMPLETE")
    print(f"  Final attacker evasion rate : {attacker.evasion_rate:.1%}")
    print(f"  Final defender accuracy     : {defender.accuracy:.1%}")
    print(f"  Total attacks generated     : {attacker.total_attacks}")
    print(f"  Attacker GRPO updates       : {attacker.update_count}")
    print(f"  Defender online updates     : {defender.update_count}")
    print(f"  Hall of Fame entries        : {len(env.hall_of_fame)}")
    print(f"  Log saved to                : {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()

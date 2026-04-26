"""
Multi-Agent Adversarial Arms Race -- Main training loop.

5 specialized attacker agents (email, chat, tool_output, document, code)
share one Gemma 3 1B base and battle a Gemma 3 4B defender simultaneously.

Each round (swarm mode):
  - All 5 agents generate attacks from their specialized vector
  - Coordinator uses UCB to focus budget on highest-evasion vectors
  - Defender classifies all attacks at once
  - Each agent updates via GRPO independently
  - Optional: 2+ agents collaborate on a distributed multi-source attack

Usage:
    HF_TOKEN=hf_xxx python arena/run_multi.py [--episodes 50] [--rounds 8] [--group 8]
"""

from __future__ import annotations
import os, sys, json, argparse, time, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from arena.multi_env import MultiAgentArmsRaceEnv, INJECTION_VECTORS
from arena.multi_attacker import MultiAgentGRPOAttacker, AGENT_CONFIGS
from arena.coordinator import AgentCoordinator
from arena.defender import LiveDefender
from arena.viz import ArenaVisualizer

_shutdown = False
def _handle_sigint(sig, frame):
    global _shutdown
    print("\n[!] CTRL+C caught -- finishing episode then saving...")
    _shutdown = True
signal.signal(signal.SIGINT, _handle_sigint)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",         type=int,   default=50)
    p.add_argument("--rounds",           type=int,   default=8)
    p.add_argument("--group",            type=int,   default=8)
    p.add_argument("--defender-adapter", type=str,   default="dataset/gemma3-4b-lora")
    p.add_argument("--save-dir",         type=str,   default="arena/checkpoints_multi")
    p.add_argument("--save-every",       type=int,   default=10)
    p.add_argument("--device",           type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-viz",           action="store_true")
    p.add_argument("--no-distributed",   action="store_true",
                   help="Disable distributed multi-source attacks")
    return p.parse_args()


def run():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("Set HF_TOKEN env var")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("  ADVERSARIAL ARMS RACE -- MULTI-AGENT SWARM")
    print(f"  Attackers: 5 specialized agents on Gemma 3 1B (GRPO)")
    print(f"  Defender : Gemma 3 4B (Online fine-tune)")
    print(f"  Episodes : {args.episodes}  Rounds/ep: {args.rounds}  Group: {args.group}")
    print(f"  Vectors  : {', '.join(INJECTION_VECTORS)}")
    print(f"  Device   : {args.device}")
    print(f"{'='*65}\n")

    env = MultiAgentArmsRaceEnv(
        episode_rounds=args.rounds,
        group_size=args.group,
        max_episodes=args.episodes,
        curriculum=True,
        swarm_mode=True,
    )

    print("Loading agents (shared base -- faster than 5 separate loads)...")
    attacker = MultiAgentGRPOAttacker(
        hf_token=hf_token,
        device=args.device,
        group_size=args.group,
    )

    coordinator = AgentCoordinator(
        agent_names=list(AGENT_CONFIGS.keys()),
        exploration=0.3,
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

    log_path = os.path.join(args.save_dir, "arms_race_multi_log.jsonl")
    log_file = open(log_path, "a")

    def log(data: dict):
        log_file.write(json.dumps(data) + "\n")
        log_file.flush()

    print("Swarm online. Arms race starting...\n")

    for ep in range(1, args.episodes + 1):
        if _shutdown:
            break

        obs = env.reset()
        context = obs["seed_context"]

        for rnd in range(args.rounds):
            if _shutdown or obs is None:
                break

            context = obs["seed_context"]
            difficulty = obs["difficulty"]
            temperature = 1.0 + difficulty * 0.5

            # --- Coordinator selects active agents ---
            active_agents = coordinator.select_agents()   # all = swarm mode

            # --- Each agent generates attacks ---
            agent_results = attacker.generate_all_agents(
                active_agents=active_agents,
                context=context,
                temperature=temperature,
            )
            # agent_results: {agent_name: (prompt, completions, full_attacks)}

            # --- Build distributed multi-source attack (2 random agents collaborate) ---
            distributed_attacks = []
            distributed_labels = []
            if not args.no_distributed and len(active_agents) >= 2 and rnd % 3 == 0:
                pair = list(active_agents)[:2]
                da_id = f"ep{ep}_rnd{rnd}"
                da = coordinator.start_distributed_attack(da_id)
                for agent_name in pair:
                    _, _, full_attacks = agent_results[agent_name]
                    fragment = full_attacks[0] if full_attacks else ""
                    coordinator.add_fragment_to_distributed(
                        da_id, agent_name, fragment[:200]
                    )
                if da.is_complete:
                    assembled = da.assemble()
                    distributed_attacks.append(assembled)
                    distributed_labels.append("injection")

            # --- Collect all attacks for defender ---
            all_attacks = []
            all_labels = []
            agent_attack_map = {}
            for agent_name in active_agents:
                _, _, full_attacks = agent_results[agent_name]
                agent_attack_map[agent_name] = full_attacks
                all_attacks.extend(full_attacks)
                all_labels.extend(["injection"] * len(full_attacks))

            all_attacks.extend(distributed_attacks)
            all_labels.extend(distributed_labels)

            # --- Defender classifies everything ---
            caught_all, results_all = defender.process_round(all_attacks, all_labels, episode=ep)

            # --- Split caught flags back per agent ---
            defender_results_per_agent = {}
            idx = 0
            for agent_name in active_agents:
                n = len(agent_attack_map[agent_name])
                defender_results_per_agent[agent_name] = caught_all[idx:idx + n]
                idx += n

            dist_caught = caught_all[idx:] if distributed_attacks else []

            # --- Per-agent GRPO updates ---
            atk_metrics_all = {}
            for agent_name in active_agents:
                prompt, completions, _ = agent_results[agent_name]
                rewards = [
                    -1.0 if c else +1.0
                    for c in defender_results_per_agent[agent_name]
                ]
                metrics = attacker.update(agent_name, prompt, completions, rewards)
                atk_metrics_all[agent_name] = metrics
                if metrics and not metrics.get("skipped"):
                    viz.feed.add_update(
                        agent_name,
                        metrics.get("pg_loss", 0),
                        attacker.stats[agent_name]["update_count"],
                    )

            # --- Coordinator update ---
            per_agent_evasion = {}
            for agent_name in active_agents:
                caught_flags = defender_results_per_agent[agent_name]
                er = sum(1 for c in caught_flags if not c) / max(1, len(caught_flags))
                per_agent_evasion[agent_name] = er
            coordinator.update_all(per_agent_evasion)

            # --- Step environment ---
            step_result = env.step_multi_round(
                agent_attacks=agent_attack_map,
                defender_results=defender_results_per_agent,
                distributed_attacks=distributed_attacks,
                distributed_caught=dist_caught,
            )
            obs = step_result["observation"]
            mr_summary = step_result["multi_round_summary"]

            # --- Feed top evasions to viz ---
            for agent_name in active_agents:
                attacks_list = agent_attack_map[agent_name]
                caught_flags = defender_results_per_agent[agent_name]
                for attack, caught in list(zip(attacks_list, caught_flags))[:2]:
                    viz.feed.add_attack(attack, caught, agent_name, ep, rnd + 1)

            # --- Render ---
            stats = env.global_stats()
            viz.render(
                episode=ep,
                round_num=rnd + 1,
                atk_evasion=attacker.evasion_rate,
                def_accuracy=defender.accuracy,
                difficulty=stats["difficulty"],
                hall_of_fame=env.hall_of_fame,
                atk_updates=sum(s["update_count"] for s in attacker.stats.values()),
                def_updates=defender.update_count,
                atk_loss=sum(
                    m.get("pg_loss", 0) for m in atk_metrics_all.values()
                    if m and not m.get("skipped")
                ) / max(1, len(atk_metrics_all)),
                def_loss=defender.loss_history[-1] if defender.loss_history else 0.0,
                total_attacks=attacker.total_attacks,
                total_rounds=stats.get("rounds_total", 0),
            )

            # --- Log ---
            log({
                "type": "round",
                "episode": ep,
                "round": rnd + 1,
                "difficulty": difficulty,
                "multi_round": mr_summary,
                "coordinator": coordinator.summary(),
                "atk_evasion": attacker.evasion_rate,
                "def_accuracy": defender.accuracy,
                "distributed_attacks": len(distributed_attacks),
            })

            if args.no_viz:
                vec_rates = coordinator.defender_weakness_profile()
                vec_str = " ".join(f"{k[:4]}={v:.0%}" for k, v in vec_rates.items())
                print(
                    f"Ep{ep:03d} Rnd{rnd+1:02d} | "
                    f"ATK {attacker.evasion_rate:.0%} DEF {defender.accuracy:.0%} | "
                    f"diff={difficulty:.2f} | {vec_str}"
                )

        # Episode log
        log({
            "type": "episode",
            "episode": ep,
            "atk_evasion": attacker.evasion_rate,
            "def_accuracy": defender.accuracy,
            "coordinator_summary": coordinator.summary(),
            "vector_hall_of_fame": {
                v: len(hof) for v, hof in env.vector_hall_of_fame.items()
            },
        })

        if ep % args.save_every == 0:
            atk_save = os.path.join(args.save_dir, f"attacker_ep{ep:04d}")
            def_save = os.path.join(args.save_dir, f"defender_ep{ep:04d}")
            attacker.save(atk_save)
            defender.save(def_save)
            print(f"[Checkpoint] Saved episode {ep}")

        if args.no_viz:
            vec_stats = env.vector_stats()
            print(f"\n{'='*55}")
            print(f"EPISODE {ep} COMPLETE")
            print(f"  Attacker evasion : {attacker.evasion_rate:.1%}")
            print(f"  Defender accuracy: {defender.accuracy:.1%}")
            print(f"  Per-vector evasion:")
            for v, rate in vec_stats.items():
                bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
                print(f"    {v:12s} [{bar}] {rate:.0%}")
            print(f"{'='*55}\n")

    viz.stop()
    log_file.close()

    attacker.save(os.path.join(args.save_dir, "attacker_final"))
    defender.save(os.path.join(args.save_dir, "defender_final"))

    print(f"\n{'='*65}")
    print("MULTI-AGENT ARMS RACE COMPLETE")
    print(f"  Total attacks generated : {attacker.total_attacks}")
    print(f"  Overall evasion rate    : {attacker.evasion_rate:.1%}")
    print(f"  Defender accuracy       : {defender.accuracy:.1%}")
    print(f"  Hall of Fame entries    : {len(env.hall_of_fame)}")
    print(f"  Per-agent stats:")
    for name, s in attacker.stats.items():
        er = s["total_evaded"] / max(1, s["total_attacks"])
        print(f"    {name:12s}: {s['total_attacks']} attacks, {er:.0%} evasion, "
              f"{s['update_count']} updates")
    print(f"  Log: {log_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run()

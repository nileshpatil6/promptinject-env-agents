"""Tests for PipelineEnv reward shaping and state machine."""

import pytest
from server.models import PromptInjectionAction
from server.pipeline_env import PipelineEnv


def _action(cls="injection", explanation="test"):
    return PromptInjectionAction(
        classification=cls,
        explanation=explanation,
        severity=0.9 if cls == "injection" else None,
    )


@pytest.fixture
def env():
    return PipelineEnv()


# ---------------------------------------------------------------------------
# Injection scenarios
# ---------------------------------------------------------------------------

class TestInjectionScenarios:
    def test_flag_at_stage_1_yields_max_reward(self, env):
        env.reset()
        result = env.step(_action("injection"))
        assert result.reward == pytest.approx(1.0)
        assert result.done is True

    def test_flag_at_stage_2_yields_0_7(self, env):
        env.reset()
        env.step(_action("benign"))  # miss stage 1
        result = env.step(_action("injection"))
        assert result.reward == pytest.approx(0.7)
        assert result.done is True

    def test_flag_at_stage_3_yields_0_4(self, env):
        env.reset()
        env.step(_action("benign"))
        env.step(_action("benign"))
        result = env.step(_action("injection"))
        assert result.reward == pytest.approx(0.4)
        assert result.done is True

    def test_flag_at_stage_4_yields_0_1(self, env):
        env.reset()
        env.step(_action("benign"))
        env.step(_action("benign"))
        env.step(_action("benign"))
        result = env.step(_action("injection"))
        assert result.reward == pytest.approx(0.1)
        assert result.done is True

    def test_injection_escapes_yields_negative_1(self, env):
        env.reset()
        rewards = []
        for _ in range(3):
            r = env.step(_action("benign"))
            rewards.append(r.reward)
        last = env.step(_action("benign"))
        rewards.append(last.reward)
        assert last.reward == pytest.approx(-1.0)
        assert last.done is True

    def test_episode_ends_after_flag(self, env):
        env.reset()
        env.step(_action("injection"))
        with pytest.raises(RuntimeError, match="Episode is done"):
            env.step(_action("injection"))


# ---------------------------------------------------------------------------
# Benign scenario (pipeline_003 is the benign one, index 2)
# ---------------------------------------------------------------------------

class TestBenignScenario:
    def test_false_positive_yields_negative_0_3(self, env):
        env._episode_num = 2  # scenario 3 = benign
        env.reset()
        result = env.step(_action("injection"))
        assert result.reward == pytest.approx(-0.3)
        assert result.done is True

    def test_correct_benign_pass_all_stages(self, env):
        env._episode_num = 2
        env.reset()
        rewards = []
        for _ in range(4):
            if env._done:
                break
            r = env.step(_action("benign"))
            rewards.append(r.reward)
        assert rewards[-1] == pytest.approx(0.5)
        assert all(r >= 0 for r in rewards)

    def test_benign_intermediate_rewards(self, env):
        env._episode_num = 2
        env.reset()
        r1 = env.step(_action("benign"))
        assert r1.reward == pytest.approx(0.1)
        assert r1.done is False


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_reset_initializes_stage_1(self, env):
        result = env.reset()
        assert result.observation["stage"] == 1
        assert result.done is False

    def test_stage_advances_on_pass(self, env):
        env.reset()
        env.step(_action("benign"))
        state = env.state()
        assert state.step == 1

    def test_episode_cycles_through_scenarios(self, env):
        for i in range(5):
            r = env.reset()
            assert r.observation is not None

    def test_observation_none_when_done(self, env):
        env.reset()
        result = env.step(_action("injection"))
        assert result.observation is None
        assert result.done is True

    def test_reward_distinct_values(self, env):
        rewards_seen = set()

        # Stage 1 flag on injection scenario (scenario 0)
        env._episode_num = 0
        env.reset()
        rewards_seen.add(env.step(_action("injection")).reward)

        # Stage 2 flag on injection scenario (scenario 0)
        env._episode_num = 0
        env.reset()
        env.step(_action("benign"))
        rewards_seen.add(env.step(_action("injection")).reward)

        # Miss all stages on injection scenario (scenario 0)
        env._episode_num = 0
        env.reset()
        for _ in range(3):
            env.step(_action("benign"))
        rewards_seen.add(env.step(_action("benign")).reward)

        assert 1.0 in rewards_seen
        assert 0.7 in rewards_seen
        assert -1.0 in rewards_seen
        assert len(rewards_seen) >= 3

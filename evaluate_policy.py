from __future__ import annotations
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from env import (
    ACTION_CONTINUE,
    ACTION_INSPECT,
    ACTION_MAINTAIN,
    EnvConfig,
    MaintenanceEnv,
)

RL_MODEL_PATH = "artifacts/rl_models/dqn_maintenance_agent.zip"
DATA_PATH = "data/ai4i2020.csv"
RL_MODEL_DIR = "artifacts/rl_models"
RISK_MODEL_PATH = "xgb_model.pkl"
COLS_PATH = "xgb_features.pkl"

def run_one_episode(env, policy_fn):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    failure_count = 0
    steps = 0
    action_list = []
    true_risk_list = []
    uncertainty_list = []

    while not done:
        action = policy_fn(obs, info)
        action_list.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        true_risk_list.append(info["true_risk"])
        uncertainty_list.append(info["initial_uncertainty"])
        total_reward += reward
        steps += 1
        if info.get("failure_event", False):
            failure_count += 1
        done = terminated or truncated

    return {
        "total_reward": total_reward,
        "steps": steps,
        "failure_count": failure_count,
        "action_list" : np.array(action_list), 
        "true_risk_list": np.array(true_risk_list), 
        "uncertainty_list": np.array(uncertainty_list)
    }


def evaluate_policy(env, policy_fn, n_episodes=100):
    results = [run_one_episode(env, policy_fn) for _ in range(n_episodes)]
    return {
        "mean_reward": float(np.mean([r["total_reward"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "failure_rate": float(np.mean([r["failure_count"] > 0 for r in results])),
        "action_list" : np.concat([r["action_list"] for r in results]),
        "true_risk_list" : np.concat([r["true_risk_list"] for r in results]), 
        "uncertainty_list" : np.concat([r["uncertainty_list"] for r in results])}


def main():
    env_config = EnvConfig(
        max_steps=20,
        inspect_cost=1.0,
        maintain_cost=3.0,
        failure_cost=12.0,
        survival_reward=1.0,
        high_risk_penalty=2.0,
        risk_threshold=0.6,
    )
    env = MaintenanceEnv(
        data_path=DATA_PATH,
        cols_path = COLS_PATH,
        risk_model_path=RISK_MODEL_PATH,
        config=env_config,
        seed=42,
    )
    rl_model = DQN.load(RL_MODEL_PATH)

    def rl_policy(obs, info):
        action, _ = rl_model.predict(obs, deterministic=True)
        return int(action)

    def always_continue(obs, info):
        return ACTION_CONTINUE

    def threshold_policy(obs, info):
        risk = obs[-1]
        if risk > 0.75:
            return ACTION_MAINTAIN
        elif risk > 0.45:
            return ACTION_INSPECT
        return ACTION_CONTINUE

    print("RL policy mean reward:", evaluate_policy(env, rl_policy, n_episodes=200)["mean_reward"])
    print("Always continue mean reward:", evaluate_policy(env, always_continue, n_episodes=200)["mean_reward"])
    print("Threshold policy mean reward:", evaluate_policy(env, threshold_policy, n_episodes=200)["mean_reward"])
    return evaluate_policy(env, rl_policy, n_episodes=200)


if __name__ == "__main__":
    main()
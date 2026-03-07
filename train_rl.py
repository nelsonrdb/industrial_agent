from __future__ import annotations

import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from risk_model import load_data, prepare_features_and_target
from env import MaintenanceEnv, EnvConfig


DATA_PATH = "data/ai4i2020.csv"
RL_MODEL_DIR = "artifacts/rl_models"
RISK_MODEL_PATH = "xgb_model.pkl"
COLS_PATH = "xgb_features.pkl"
os.makedirs(RL_MODEL_DIR, exist_ok=True)


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
    env = Monitor(env, filename="artifacts/monitor.csv")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=20_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0,
        tensorboard_log="artifacts/tensorboard/",
        seed=42,
    )

    model.learn(total_timesteps=100000, progress_bar=True)

    save_path = os.path.join(RL_MODEL_DIR, "dqn_maintenance_agent")
    model.save(save_path)

    print(f"Saved RL model to: {save_path}")


if __name__ == "__main__":
    main()
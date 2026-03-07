from __future__ import annotations

import random
from dataclasses import dataclass

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
from gymnasium import spaces
from risk_model import load_data, prepare_features_and_target


ACTION_CONTINUE = 0
ACTION_INSPECT = 1
ACTION_MAINTAIN = 2


@dataclass
class EnvConfig:
    max_steps: int = 20
    inspect_cost: float = 1.0
    maintain_cost: float = 3.0
    failure_cost: float = 12.0
    survival_reward: float = 1.0
    high_risk_penalty: float = 2.0
    risk_threshold: float = 0.6
    wear_increase_min: float = 3.0
    wear_increase_max: float = 10.0
    maintain_reset_min: float = 0.0
    maintain_reset_max: float = 15.0
    noise_std_temp: float = 0.3
    noise_std_speed: float = 8.0
    noise_std_torque: float = 0.8


class MaintenanceEnv(gym.Env):
    """
    Observation:
        ['Air temperature',
        'Process temperature',
        'Rotational speed',
        'Torque',
        'Tool wear',
        'Type_L',
        'Type_M',
        'temp_diff',
        'torque_speed_ratio',
        'wear_rate']

    Actions:
        0 = continue
        1 = inspect
        2 = maintain
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str,
        cols_path: str,
        risk_model_path: str,
        config: EnvConfig | None = None,
        seed: int = 42,):
        super().__init__()
        self.data_path = data_path
        self.feature_cols = joblib.load(cols_path)
        self.df, _, _ = prepare_features_and_target(load_data(self.data_path))
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.config = config or EnvConfig()
        self.risk_model = joblib.load(risk_model_path)


        self.action_space = spaces.Discrete(3)

        #Approximative Boundaries
        low = np.array([250, 250, 1000, 0, 0, 0, 0, -100, 0, 0], dtype=np.float32)

        high = np.array([350, 350, 3000, 100, 300, 1, 1, 100, 0.1, 0.3], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_features: dict | None = None
        self.current_step = 0

    def _sample_initial_state(self) -> dict:
        return self.df.sample(n=1, random_state=int(self.rng.integers(0, 1_000_000))).iloc[0].to_dict()

    def _clip_features(self, features: dict) -> dict:
        features["Air temperature"] = float(np.clip(features["Air temperature"], 250, 350))
        features["Process temperature"] = float(np.clip(features["Process temperature"], 250, 350))
        features["Rotational speed"] = float(np.clip(features["Rotational speed"], 1000, 3000))
        features["Torque"] = float(np.clip(features["Torque"], 0, 100))
        features["Tool wear"] = float(np.clip(features["Tool wear"], 0, 300))

        features["temp_diff"] = features["Process temperature"] - features["Air temperature"]
        features["torque_speed_ratio"] = features["Torque"] / features["Rotational speed"]
        features["wear_rate"] = features["Tool wear"] / features["Rotational speed"]
        return features

    def _predict_risk(self, features: dict) -> float:
        x = pd.DataFrame([features], columns=self.feature_cols)
        risk = self.risk_model.predict_proba(x)[:, 1][0]
        return float(np.clip(risk, 0.0, 1.0))

    def _to_obs(self, features: dict) -> np.ndarray:
        obs = np.array(
            [
                features["Air temperature"],
                features["Process temperature"],
                features["Rotational speed"],
                features["Torque"],
                features["Tool wear"],
                float(features["Type_L"]),
                float(features["Type_M"]),
                features["temp_diff"],
                features["torque_speed_ratio"],
                features["wear_rate"],
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_features = self._sample_initial_state()
        self.current_features = self._clip_features(self.current_features)

        risk_score = self._predict_risk(self.current_features)
        obs = self._to_obs(self.current_features)

        info = {
            "risk_score": risk_score,
            "step": self.current_step,
        }
        return obs, info

    def step(self, action: int):
        assert self.current_features is not None, "Call reset() before step()."
        cfg = self.config
        self.current_step += 1

        features = self.current_features.copy()
        reward = 0.0

        # Action dynamics
        if action == ACTION_CONTINUE:
            reward += cfg.survival_reward

            features["Tool wear"] += float(
                self.rng.uniform(cfg.wear_increase_min, cfg.wear_increase_max)
            )
            features["Air temperature"] += float(self.rng.normal(0, cfg.noise_std_temp))
            features["Process temperature"] += float(self.rng.normal(0, cfg.noise_std_temp))
            features["Rotational speed"] += float(self.rng.normal(0, cfg.noise_std_speed))
            features["Torque"] += float(self.rng.normal(0, cfg.noise_std_torque))

        elif action == ACTION_INSPECT:
            #smaller impact than continue, but small cost
            reward -= cfg.inspect_cost
            features["Tool wear"] += float(self.rng.uniform(0.5, 2.0))
            features["Air temperature"] += float(self.rng.normal(0, cfg.noise_std_temp * 0.3))
            features["Process temperature"] += float(self.rng.normal(0, cfg.noise_std_temp * 0.3))
            features["Rotational speed"] += float(self.rng.normal(0, cfg.noise_std_speed * 0.2))
            features["Torque"] += float(self.rng.normal(0, cfg.noise_std_torque * 0.2))

        elif action == ACTION_MAINTAIN:
            reward -= cfg.maintain_cost

            # Partial wear reset
            features["Tool wear"] = float(
                self.rng.uniform(cfg.maintain_reset_min, cfg.maintain_reset_max)
            )
            # Small stabilisation
            features["Torque"] += float(self.rng.normal(-1.0, 0.5))
            features["Rotational speed"] += float(self.rng.normal(0.0, 5.0))

        else:
            raise ValueError(f"Invalid action: {action}")

        features = self._clip_features(features)
        risk_score = self._predict_risk(features)

        # Penalty if we continnue as risk is high
        if action == ACTION_CONTINUE and risk_score > cfg.risk_threshold:
            reward -= cfg.high_risk_penalty * risk_score

        # Failure event
        failure_event = self.rng.random() < risk_score
        terminated = False
        truncated = False

        if failure_event:
            reward -= cfg.failure_cost
            terminated = True

        if self.current_step >= cfg.max_steps:
            truncated = True

        self.current_features = features
        obs = self._to_obs(features)

        info = {
            "risk_score": risk_score,
            "failure_event": failure_event,
            "step": self.current_step,
            "action_name": ["continue", "inspect", "maintain"][action],
        }

        return obs, float(reward), terminated, truncated, info
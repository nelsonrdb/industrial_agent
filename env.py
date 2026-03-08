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
    max_steps: int = 40
    inspect_cost: float = 0.2
    maintain_cost: float = 0.5
    failure_cost: float = 18.0
    survival_reward: float = 1.5
    high_risk_penalty: float = 4.0
    risk_threshold: float = 0.55
    wear_increase_min: float = 10.0
    wear_increase_max: float = 50.0
    maintain_reset_min: float = 0.0
    maintain_reset_max: float = 8.0
    noise_std_temp: float = 0.3
    noise_std_speed: float = 8.0
    noise_std_torque: float = 0.8

    uncertainty_min: float = 0.02
    uncertainty_max: float = 0.50
    uncertainty_increase: float = 0.04
    uncertainty_decrease: float = 0.20
    uncertainty_noise: float = 0.02


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
        self.observed_risk = 0.0
        self.uncertainty = 0.15

        self.action_space = spaces.Discrete(3)

        #Approximative Boundaries
        low = np.array([250, 250, 1000, 0, 0, 0, 0, -100, 0, 0, 0, 0.0], dtype=np.float32)
        high = np.array([350, 350, 3000, 100, 300, 1, 1, 100, 0.1, 0.3, 1, 0.2], dtype=np.float32)

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
    
    def _observe_risk(self, true_risk: float, inspected: bool) -> tuple[float, float]:
        cfg = self.config

        if inspected:
            self.uncertainty -= cfg.uncertainty_decrease
        else:
            self.uncertainty += cfg.uncertainty_increase

        # petit côté stochastique
        self.uncertainty += float(self.rng.normal(0.0, cfg.uncertainty_noise))

        # borne l'incertitude
        self.uncertainty = float(
            np.clip(self.uncertainty, cfg.uncertainty_min, cfg.uncertainty_max)
        )

        observed = np.clip(
            true_risk + self.rng.normal(0.0, self.uncertainty),
            0.0,
            1.0
        )

        return float(observed), float(self.uncertainty)
    
    def _to_obs(self, features: dict, observe_risk:float, uncertainty: float) -> np.ndarray:
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
                observe_risk, 
                uncertainty
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_features = self._sample_initial_state()
        self.current_features = self._clip_features(self.current_features)

        self.true_risk = self._predict_risk(self.current_features)
        self.uncertainty = 0.08
        self.observed_risk = float(
            np.clip(self.true_risk + self.rng.normal(0.0, self.uncertainty), 0.0, 1.0)
        )
        obs = self._to_obs(self.current_features, self.observed_risk, self.uncertainty)

        info = {
            "true_risk": self.true_risk,
            "observed_risk": self.observed_risk,
            "uncertainty": self.uncertainty,
            "step": self.current_step,
        }
        return obs, info

    def step(self, action: int):
        assert self.current_features is not None, "Call reset() before step()."
        initial_uncertainty = self.uncertainty
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
            features["Air temperature"] += float(self.rng.normal(2, cfg.noise_std_temp))
            features["Process temperature"] += float(self.rng.normal(2, cfg.noise_std_temp))
            features["Rotational speed"] += float(self.rng.normal(2, cfg.noise_std_speed))
            features["Torque"] += float(self.rng.normal(2, cfg.noise_std_torque))

        elif action == ACTION_INSPECT:
            #smaller impact than continue, but small cost
            reward -= cfg.inspect_cost
            features["Tool wear"] += float(
                self.rng.uniform(cfg.wear_increase_min, cfg.wear_increase_max)
            )
            features["Air temperature"] += float(self.rng.normal(-20, cfg.noise_std_temp))
            features["Process temperature"] += float(self.rng.normal(-20, cfg.noise_std_temp))
            features["Rotational speed"] += float(self.rng.normal(-80, cfg.noise_std_speed))
            features["Torque"] += float(self.rng.normal(-3, cfg.noise_std_torque))

        elif action == ACTION_MAINTAIN:
            reward -= cfg.maintain_cost

            features["Tool wear"] = float(
                self.rng.uniform(cfg.maintain_reset_min, cfg.maintain_reset_max)
            )
            features["Torque"] += float(self.rng.normal(-100, 0.3))
            features["Rotational speed"] += float(self.rng.normal(-500, 3.0))
            features["Process temperature"] += float(self.rng.normal(-100, cfg.noise_std_temp))

        else:
            raise ValueError(f"Invalid action: {action}")

        features = self._clip_features(features)
        true_risk = self._predict_risk(features)

        if action == ACTION_INSPECT:
            observed_risk, uncertainty = self._observe_risk(true_risk, inspected=True)
        else:
            observed_risk, uncertainty = self._observe_risk(true_risk, inspected=False)

        if action == ACTION_CONTINUE and true_risk > 0.55:
            reward -= 4.0 * true_risk

        if action == ACTION_MAINTAIN and true_risk < 0.30:
            reward -= 2.0

        if action != ACTION_MAINTAIN and true_risk > 0.70:
            reward -= 8.0
        

        if 0.30 <= true_risk <= 0.60 and action != ACTION_INSPECT:
            reward -= 5

        if 0.30 <= true_risk <= 0.60 and action == ACTION_INSPECT:
            reward += 4

        # Failure event
        failure_event = self.rng.random() < true_risk
        terminated = False
        truncated = False

        if failure_event:
            reward -= cfg.failure_cost
            terminated = True

        if self.current_step >= cfg.max_steps:
            truncated = True

        self.current_features = features
        obs = self._to_obs(features, observed_risk, uncertainty)

        info = {
            "true_risk": true_risk,
            "observed_risk": observed_risk,
            "initial_uncertainty": initial_uncertainty,
            "failure_event": failure_event,
            "step": self.current_step,
            "action_name": ["continue", "inspect", "maintain"][action],
        }

        return obs, float(reward), terminated, truncated, info
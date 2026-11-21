
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SmartLearnEnv(gym.Env):
    """
    A simulated adaptive English-learning environment.

    The agent selects learning activities for a simulated primary-school
    learner. Learner skills, confidence, engagement and fatigue persist
    and change throughout an episode.
    """

    metadata = {"render_modes": []}

    ACTIONS = {
        0: "Vocabulary introduction",
        1: "Listening practice",
        2: "Local-language support scaffold",
        3: "Word matching",
        4: "Reading comprehension",
        5: "Sentence building",
        6: "Revision of weakest skill",
    }

    SKILLS = [
        "vocabulary",
        "listening",
        "comprehension",
        "sentence_building",
    ]

    PROFILES = {
        "beginner": [0.15, 0.12, 0.10, 0.08],
        "developing": [0.35, 0.30, 0.28, 0.25],
        "intermediate": [0.60, 0.55, 0.50, 0.48],
        "uneven": [0.65, 0.25, 0.55, 0.20],
    }

    def __init__(self, max_steps=20, render_mode=None):
        super().__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode

        # Seven possible learning activities.
        self.action_space = spaces.Discrete(7)

        # Eleven normalized observations, each between 0 and 1.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32,
        )

        self.state = None
        self.steps = 0
        self.last_action = None
        self.repeat_count = 0
        self.mistake_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        profile = options.get("profile", "beginner")

        if profile not in self.PROFILES:
            raise ValueError(
                f"Unknown learner profile: {profile}. "
                f"Choose from {list(self.PROFILES)}."
            )

        initial_skills = np.array(
            self.PROFILES[profile], dtype=np.float32
        )

        self.state = {
            "skills": initial_skills.copy(),
            "confidence": 0.40,
            "engagement": 0.75,
            "fatigue": 0.05,
            "difficulty": 0.45,
            "previous_success": 0.0,
        }

        self.steps = 0
        self.last_action = None
        self.repeat_count = 0
        self.mistake_history = []

        observation = self._get_observation()

        info = {
            "profile": profile,
            "skills": self._skill_dictionary(),
        }

        return observation, info

    def _get_observation(self):
        skills = self.state["skills"]

        remaining_time = (
            self.max_steps - self.steps
        ) / self.max_steps

        recent_mistakes = (
            np.mean(self.mistake_history[-5:])
            if self.mistake_history
            else 0.0
        )

        observation = np.array(
            [
                skills[0],
                skills[1],
                skills[2],
                skills[3],
                self.state["confidence"],
                self.state["engagement"],
                self.state["fatigue"],
                self.state["difficulty"],
                self.state["previous_success"],
                remaining_time,
                recent_mistakes,
            ],
            dtype=np.float32,
        )

        return np.clip(observation, 0.0, 1.0)

    def _skill_dictionary(self):
        return {
            name: round(float(value), 3)
            for name, value in zip(
                self.SKILLS, self.state["skills"]
            )
        }

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1

        # Track repeated activities.
        if action == self.last_action:
            self.repeat_count += 1
        else:
            self.repeat_count = 0

        self.last_action = action

        skills = self.state["skills"]
        old_skills = skills.copy()
        old_confidence = self.state["confidence"]
        old_engagement = self.state["engagement"]
        old_fatigue = self.state["fatigue"]

        # Select which skill the activity primarily targets.
        if action in (2, 6):
            target = int(np.argmin(skills))
        else:
            target = {
                0: 0,
                1: 1,
                3: 0,
                4: 2,
                5: 3,
            }[action]

        # The learner's chance of success depends on their state.
        success_probability = (
            0.55 * skills[target]
            + 0.15 * self.state["confidence"]
            + 0.10 * self.state["engagement"]
            + 0.20 * (1.0 - self.state["difficulty"])
            - 0.15 * self.state["fatigue"]
        )

        # Scaffolding makes the activity more accessible.
        if action == 2:
            success_probability += 0.15

        success_probability = float(
            np.clip(success_probability, 0.05, 0.95)
        )

        success = bool(
            self.np_random.random() < success_probability
        )

        # Learning gains persist after every activity.
        if success:
            skills[target] += (
                0.035 * (0.7 + 0.6 * self.state["difficulty"])
            )
            self.state["confidence"] += 0.03
            self.state["engagement"] += 0.015
            self.state["difficulty"] += 0.015
        else:
            skills[target] += 0.012
            self.state["confidence"] -= 0.015
            self.state["engagement"] -= 0.025
            self.state["difficulty"] -= 0.025

        # Scaffolding and revision are less demanding activities.
        if action == 2:
            self.state["difficulty"] -= 0.025

        # Fatigue accumulates as the learning session continues.
        fatigue_increase = {
            0: 0.045,
            1: 0.045,
            2: 0.020,
            3: 0.040,
            4: 0.055,
            5: 0.055,
            6: 0.025,
        }[action]

        self.state["fatigue"] += fatigue_increase

        # Repeated activities can reduce engagement.
        if self.repeat_count >= 2:
            self.state["engagement"] -= 0.04

        # Keep all learner variables within their valid ranges.
        skills[:] = np.clip(skills, 0.0, 1.0)

        self.state["confidence"] = float(
            np.clip(self.state["confidence"], 0.0, 1.0)
        )
        self.state["engagement"] = float(
            np.clip(self.state["engagement"], 0.0, 1.0)
        )
        self.state["fatigue"] = float(
            np.clip(self.state["fatigue"], 0.0, 1.0)
        )
        self.state["difficulty"] = float(
            np.clip(self.state["difficulty"], 0.0, 1.0)
        )

        self.state["previous_success"] = float(success)
        self.mistake_history.append(0 if success else 1)

        # Reward combines learning progress and learner wellbeing.
        learning_improvement = float(
            np.sum(skills - old_skills)
        )

        reward = (
            10.0 * learning_improvement
            + (1.0 if success else -0.2)
            + 2.0 * (
                self.state["confidence"] - old_confidence
            )
            + 2.0 * (
                self.state["engagement"] - old_engagement
            )
            - 1.5 * (
                self.state["fatigue"] - old_fatigue
            )
        )

        # Discourage repeatedly presenting the same activity.
        if self.repeat_count >= 2:
            reward -= 0.25

        # Episode termination conditions.
        mastery_reached = bool(np.all(skills >= 0.90))
        excessive_fatigue = self.state["fatigue"] >= 0.95

        terminated = mastery_reached or excessive_fatigue
        truncated = self.steps >= self.max_steps and not terminated

        observation = self._get_observation()

        info = {
            "activity": self.ACTIONS[action],
            "success": success,
            "success_probability": round(
                success_probability, 3
            ),
            "target_skill": self.SKILLS[target],
            "skills": self._skill_dictionary(),
            "mean_mastery": round(float(np.mean(skills)), 3),
            "confidence": round(
                self.state["confidence"], 3
            ),
            "engagement": round(
                self.state["engagement"], 3
            ),
            "fatigue": round(
                self.state["fatigue"], 3
            ),
            "steps": self.steps,
            "termination_reason": (
                "mastery reached"
                if mastery_reached
                else "excessive fatigue"
                if excessive_fatigue
                else "time limit"
                if truncated
                else "ongoing"
            ),
        }

        return observation, float(reward), terminated, truncated, info

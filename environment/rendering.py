
"""
SmartLearn Ghana - Pygame Interface
Mission-Based Reinforcement Learning Simulation

Run through play.py. The renderer reads the existing environment state
and displays learner progress, policy decisions, and episode feedback.
"""

import math
import pygame


# ---------------------------------------------------------------------
# COLOUR PALETTE
# ---------------------------------------------------------------------

BG = (9, 15, 30)
PANEL = (17, 26, 46)
PANEL_ALT = (22, 34, 57)
BORDER = (38, 54, 79)

WHITE = (239, 245, 255)
MUTED = (139, 157, 184)
SUBTLE = (91, 111, 141)

BLUE = (69, 139, 255)
CYAN = (57, 211, 224)
MINT = (73, 222, 169)
PURPLE = (165, 126, 255)
AMBER = (255, 190, 92)
RED = (255, 105, 117)

SKILL_COLOURS = [BLUE, CYAN, PURPLE, MINT]

WIDTH = 1100
HEIGHT = 700


# ---------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------

def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def percentage(value):
    return f"{clamp(value) * 100:.0f}%"


def draw_text(surface, text, font, colour, position):
    image = font.render(str(text), True, colour)
    surface.blit(image, position)
    return image.get_rect(topleft=position)


def draw_panel(surface, rect, colour=PANEL, border=BORDER, radius=15):
    pygame.draw.rect(surface, colour, rect, border_radius=radius)
    if border:
        pygame.draw.rect(
            surface, border, rect, width=1, border_radius=radius
        )


def draw_track(surface, x, y, width, height, value, colour):
    """Draw a clean horizontal progress track."""
    pygame.draw.rect(
        surface, (35, 49, 72),
        (x, y, width, height),
        border_radius=height // 2
    )

    filled = int(width * clamp(value))
    if filled > 0:
        pygame.draw.rect(
            surface, colour,
            (x, y, max(height, filled), height),
            border_radius=height // 2
        )


def draw_dot(surface, x, y, colour, radius=4):
    pygame.draw.circle(surface, colour, (int(x), int(y)), radius)


# ---------------------------------------------------------------------
# MAIN RENDERER
# ---------------------------------------------------------------------

class PygameRenderer:

    def __init__(self, width=WIDTH, height=HEIGHT):
        pygame.init()
        pygame.display.set_caption(
            "SmartLearn Ghana | Adaptive Learning Intelligence"
        )

        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.fonts = {
            "brand": pygame.font.SysFont("segoeui", 23, bold=True),
            "heading": pygame.font.SysFont("segoeui", 19, bold=True),
            "body": pygame.font.SysFont("segoeui", 15),
            "small": pygame.font.SysFont("segoeui", 12),
            "tiny": pygame.font.SysFont("segoeui", 10),
            "number": pygame.font.SysFont("segoeui", 27, bold=True),
            "large": pygame.font.SysFont("segoeui", 34, bold=True),
        }

        self.mastery_history = []
        self.last_recorded_step = None
        self.last_episode = None
        self.current_agent = "TRAINED POLICY"

        self.running = True

    # -----------------------------------------------------------------
    # ENVIRONMENT DATA
    # -----------------------------------------------------------------

    def _get_state(self, env):
        state = getattr(env, "state", None)

        if state is None and hasattr(env, "unwrapped"):
            state = getattr(env.unwrapped, "state", None)

        return state if isinstance(state, dict) else {}

    def _get_skills(self, env, state):
        names = getattr(env, "SKILLS", None)
        values = state.get("skills", [])

        try:
            values = list(values)
        except TypeError:
            values = []

        if names is None:
            names = [
                "Vocabulary",
                "Listening",
                "Comprehension",
                "Sentence building",
            ]

        names = list(names)

        result = []
        for index, name in enumerate(names):
            value = values[index] if index < len(values) else 0.0
            result.append((str(name).replace("_", " ").title(), clamp(value)))

        return result[:4]

    def _get_action_name(self, env, info):
        action = info.get("activity")

        if action:
            return str(action).replace("_", " ").title()

        action_index = info.get("action")

        if action_index is not None:
            actions = getattr(env, "ACTIONS", {})

            if isinstance(actions, dict):
                action_name = actions.get(action_index)
                if action_name is not None:
                    return str(action_name).replace("_", " ").title()

            elif isinstance(actions, (list, tuple)):
                if 0 <= action_index < len(actions):
                    return str(actions[action_index]).replace("_", " ").title()

        return "Waiting for next decision"

    def _record_mastery(self, info):
        step = info.get("steps")
        episode = info.get("episode", 0)
        mastery = info.get("mean_mastery")

        if mastery is None:
            skills = info.get("skills", {})
            if isinstance(skills, dict) and skills:
                mastery = sum(skills.values()) / len(skills)

        if mastery is None:
            return

        if episode != self.last_episode:
            self.mastery_history = []
            self.last_recorded_step = None
            self.last_episode = episode

        if step != self.last_recorded_step:
            self.mastery_history.append(clamp(mastery))
            self.mastery_history = self.mastery_history[-40:]
            self.last_recorded_step = step

    # -----------------------------------------------------------------
    # HEADER
    # -----------------------------------------------------------------

    def _draw_header(self, agent_name, baseline, paused):
        screen = self.screen

        pygame.draw.rect(screen, (12, 21, 39), (0, 0, WIDTH, 76))
        pygame.draw.line(screen, BORDER, (0, 75), (WIDTH, 75), 1)

        # Brand mark
        pygame.draw.rect(
            screen, BLUE, (25, 19, 36, 36), border_radius=10
        )
        draw_text(
            screen, "S", self.fonts["heading"], WHITE, (37, 25)
        )

        draw_text(
            screen, "SMARTLEARN",
            self.fonts["brand"], WHITE, (73, 17)
        )
        draw_text(
            screen, "GHANA",
            self.fonts["small"], CYAN, (75, 45)
        )

        pygame.draw.line(
            screen, BORDER, (190, 19), (190, 56), 1
        )

        draw_text(
            screen, "ADAPTIVE LEARNING SYSTEM",
            self.fonts["small"], MUTED, (210, 22)
        )
        draw_text(
            screen, "MISSION-BASED REINFORCEMENT LEARNING",
            self.fonts["tiny"], SUBTLE, (210, 43)
        )

        # Agent status
        label = "RANDOM BASELINE" if baseline else agent_name
        label = label.upper()

        pygame.draw.rect(
            screen, (28, 48, 76), (770, 22, 190, 32),
            border_radius=8
        )
        draw_dot(screen, 786, 38, AMBER if baseline else MINT, 4)

        draw_text(
            screen, label,
            self.fonts["small"], WHITE, (798, 29)
        )

        status_colour = AMBER if paused else MINT
        status_text = "PAUSED" if paused else "LIVE SIMULATION"

        draw_dot(screen, 984, 38, status_colour, 4)
        draw_text(
            screen, status_text,
            self.fonts["tiny"], status_colour, (996, 32)
        )

    # -----------------------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------------------

    def _draw_metrics(self, info):
        cards = [
            ("MEAN MASTERY", info.get("mean_mastery", 0), BLUE),
            ("CONFIDENCE", info.get("confidence", 0), PURPLE),
            ("ENGAGEMENT", info.get("engagement", 0), MINT),
            ("FATIGUE", info.get("fatigue", 0), AMBER),
        ]

        card_width = 249
        start_x = 25
        gap = 12
        y = 91

        for index, (label, value, colour) in enumerate(cards):
            x = start_x + index * (card_width + gap)

            draw_panel(
                self.screen,
                (x, y, card_width, 76),
                PANEL
            )

            draw_text(
                self.screen, label,
                self.fonts["tiny"], MUTED, (x + 16, y + 13)
            )

            draw_text(
                self.screen, percentage(value),
                self.fonts["number"], WHITE, (x + 16, y + 32)
            )

            draw_track(
                self.screen, x + 133, y + 43,
                96, 7, value, colour
            )

    # -----------------------------------------------------------------
    # LEARNER PROGRESS PANEL
    # -----------------------------------------------------------------

    def _draw_learner_panel(self, skills, state):
        x, y, w, h = 25, 183, 520, 280

        draw_panel(self.screen, (x, y, w, h))

        draw_text(
            self.screen, "LEARNER DEVELOPMENT",
            self.fonts["heading"], WHITE, (x + 20, y + 17)
        )
        draw_text(
            self.screen, "Simulated learner | Skill mastery",
            self.fonts["small"], MUTED, (x + 20, y + 44)
        )

        # Four skill progress rows
        for index, (name, value) in enumerate(skills):
            row_y = y + 91 + index * 43
            colour = SKILL_COLOURS[index % len(SKILL_COLOURS)]

            draw_dot(self.screen, x + 24, row_y + 6, colour, 4)

            draw_text(
                self.screen, name,
                self.fonts["body"], WHITE, (x + 38, row_y - 3)
            )

            draw_text(
                self.screen, percentage(value),
                self.fonts["small"], colour, (x + 454, row_y - 3)
            )

            draw_track(
                self.screen, x + 38, row_y + 19,
                436, 7, value, colour
            )

        # Learner wellbeing strip
        pygame.draw.line(
            self.screen, BORDER,
            (x + 20, y + 253), (x + w - 20, y + 253), 1
        )

        difficulty = clamp(state.get("difficulty", 0.5))

        draw_text(
            self.screen, "CURRENT DIFFICULTY",
            self.fonts["tiny"], MUTED, (x + 20, y + 260)
        )
        draw_text(
            self.screen, percentage(difficulty),
            self.fonts["small"], AMBER, (x + 175, y + 257)
        )

    # -----------------------------------------------------------------
    # AI DECISION PANEL
    # -----------------------------------------------------------------

    def _draw_decision_panel(self, env, info, baseline):
        x, y, w, h = 560, 183, 515, 280

        draw_panel(self.screen, (x, y, w, h))

        draw_text(
            self.screen, "POLICY DECISION",
            self.fonts["heading"], WHITE, (x + 20, y + 17)
        )
        draw_text(
            self.screen, "The agent's selected learning activity",
            self.fonts["small"], MUTED, (x + 20, y + 44)
        )

        action_name = self._get_action_name(env, info)

        # Decision highlight
        pygame.draw.rect(
            self.screen, (25, 46, 76),
            (x + 20, y + 78, w - 40, 82),
            border_radius=12
        )

        pygame.draw.rect(
            self.screen, BLUE,
            (x + 20, y + 78, 4, 82),
            border_radius=2
        )

        draw_text(
            self.screen, "SELECTED ACTIVITY",
            self.fonts["tiny"], CYAN, (x + 40, y + 91)
        )

        # Wrap activity name if necessary
        action_surface = self.fonts["heading"].render(
            action_name, True, WHITE
        )

        if action_surface.get_width() > w - 90:
            action_name = action_name[:34] + "..."

        draw_text(
            self.screen, action_name,
            self.fonts["heading"], WHITE, (x + 40, y + 113)
        )

        success = bool(info.get("success", False))
        probability = clamp(info.get("success_probability", 0))

        draw_text(
            self.screen, "PREDICTED SUCCESS PROBABILITY",
            self.fonts["tiny"], MUTED, (x + 22, y + 183)
        )

        draw_text(
            self.screen, percentage(probability),
            self.fonts["number"], MINT if probability >= 0.5 else AMBER,
            (x + 22, y + 204)
        )

        draw_track(
            self.screen, x + 155, y + 221,
            315, 9, probability, MINT if probability >= 0.5 else AMBER
        )

        # Outcome
        outcome_text = "ACTIVITY SUCCESSFUL" if success else "ACTIVITY NOT SUCCESSFUL"
        outcome_colour = MINT if success else RED

        draw_dot(self.screen, x + 29, y + 255, outcome_colour, 4)
        draw_text(
            self.screen, outcome_text,
            self.fonts["small"], outcome_colour, (x + 42, y + 247)
        )

        if baseline:
            draw_text(
                self.screen, "Random action selection",
                self.fonts["tiny"], MUTED, (x + 290, y + 250)
            )
        else:
            draw_text(
                self.screen, "Learned policy",
                self.fonts["tiny"], MUTED, (x + 340, y + 250)
            )

    # -----------------------------------------------------------------
    # LEARNING CURVE
    # -----------------------------------------------------------------

    def _draw_learning_curve(self):
        x, y, w, h = 25, 480, 650, 185

        draw_panel(self.screen, (x, y, w, h))

        draw_text(
            self.screen, "MASTERY TRAJECTORY",
            self.fonts["heading"], WHITE, (x + 20, y + 15)
        )
        draw_text(
            self.screen, "Mean skill mastery across this episode",
            self.fonts["small"], MUTED, (x + 20, y + 41)
        )

        chart_x = x + 52
        chart_y = y + 72
        chart_w = w - 82
        chart_h = 82

        # Grid and axis labels
        for i in range(5):
            fraction = i / 4
            line_y = chart_y + int(chart_h * fraction)

            pygame.draw.line(
                self.screen, (39, 53, 76),
                (chart_x, line_y),
                (chart_x + chart_w, line_y), 1
            )

            label = f"{100 - i * 25}%"
            draw_text(
                self.screen, label,
                self.fonts["tiny"], SUBTLE,
                (x + 12, line_y - 6)
            )

        if len(self.mastery_history) >= 2:
            points = []

            for index, value in enumerate(self.mastery_history):
                px = chart_x + int(
                    index / max(1, len(self.mastery_history) - 1) * chart_w
                )
                py = chart_y + chart_h - int(value * chart_h)
                points.append((px, py))

            pygame.draw.lines(
                self.screen, BLUE, False, points, 3
            )

            for px, py in points[-5:]:
                pygame.draw.circle(
                    self.screen, CYAN, (px, py), 3
                )

            # Current value
            latest = self.mastery_history[-1]
            draw_text(
                self.screen, f"CURRENT {percentage(latest)}",
                self.fonts["tiny"], CYAN, (x + 500, y + 19)
            )
        else:
            draw_text(
                self.screen, "Collecting episode data...",
                self.fonts["small"], MUTED,
                (chart_x + 140, chart_y + 29)
            )

        draw_text(
            self.screen, "EARLIER STEPS",
            self.fonts["tiny"], SUBTLE, (chart_x, y + 161)
        )
        draw_text(
            self.screen, "LATEST STEP",
            self.fonts["tiny"], SUBTLE,
            (chart_x + chart_w - 80, y + 161)
        )

    # -----------------------------------------------------------------
    # SESSION PANEL
    # -----------------------------------------------------------------

    def _draw_session_panel(self, info, paused):
        x, y, w, h = 690, 480, 385, 185

        draw_panel(self.screen, (x, y, w, h))

        draw_text(
            self.screen, "SESSION MONITOR",
            self.fonts["heading"], WHITE, (x + 20, y + 15)
        )

        step = int(info.get("steps", 0))
        max_steps = 20
        reward = float(info.get("reward", 0))

        metrics = [
            ("EPISODE STEP", f"{step} / {max_steps}"),
            ("LATEST REWARD", f"{reward:+.2f}"),
            ("SESSION STATUS", "PAUSED" if paused else "RUNNING"),
        ]

        for index, (label, value) in enumerate(metrics):
            row_y = y + 57 + index * 34

            draw_text(
                self.screen, label,
                self.fonts["tiny"], MUTED, (x + 20, row_y)
            )

            colour = AMBER if label == "SESSION STATUS" and paused else WHITE

            draw_text(
                self.screen, value,
                self.fonts["body"], colour, (x + 225, row_y - 3)
            )

        pygame.draw.line(
            self.screen, BORDER,
            (x + 20, y + 157), (x + w - 20, y + 157), 1
        )

        draw_text(
            self.screen, "SPACE  Pause / Resume",
            self.fonts["tiny"], MUTED, (x + 20, y + 165)
        )
        draw_text(
            self.screen, "R  Restart",
            self.fonts["tiny"], MUTED, (x + 205, y + 165)
        )

    # -----------------------------------------------------------------
    # DRAW FRAME
    # -----------------------------------------------------------------

    def draw(
        self,
        env,
        info=None,
        baseline=False,
        paused=False,
        agent_name=None,
        episode=1,
        **kwargs
    ):
        if info is None:
            info = {}

        if agent_name:
            self.current_agent = str(agent_name).upper()

        state = self._get_state(env)
        skills = self._get_skills(env, state)

        self._record_mastery(info)

        self.screen.fill(BG)

        self._draw_header(
            self.current_agent,
            baseline,
            paused
        )

        self._draw_metrics(info)

        self._draw_learner_panel(skills, state)

        self._draw_decision_panel(
            env, info, baseline
        )

        self._draw_learning_curve()

        self._draw_session_panel(info, paused)

        pygame.display.flip()
        self.clock.tick(30)

    # -----------------------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------------------

    def close(self):
        pygame.quit()


# Compatibility aliases in case another project file uses these names.
SmartLearnRenderer = PygameRenderer
EnvironmentRenderer = PygameRenderer
Renderer = PygameRenderer

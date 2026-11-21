
import pygame


class SmartLearnRenderer:
    """Visual interface for the SmartLearn Ghana environment."""

    WIDTH = 1100
    HEIGHT = 760

    BACKGROUND = (17, 24, 39)
    PANEL = (31, 41, 55)
    PANEL_LIGHT = (42, 55, 72)
    WHITE = (245, 247, 250)
    MUTED = (160, 174, 192)
    BLUE = (75, 145, 255)
    GREEN = (48, 190, 130)
    ORANGE = (245, 166, 65)
    RED = (235, 90, 90)

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT)
        )

        pygame.display.set_caption(
            "SmartLearn Ghana | Adaptive English Learning"
        )

        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont(
            "arial", 30, bold=True
        )
        self.heading_font = pygame.font.SysFont(
            "arial", 19, bold=True
        )
        self.body_font = pygame.font.SysFont("arial", 16)
        self.small_font = pygame.font.SysFont("arial", 13)

    def text(self, message, x, y, font=None, color=None):
        font = font or self.body_font
        color = color or self.WHITE

        surface = font.render(
            str(message), True, color
        )

        self.screen.blit(surface, (x, y))

    def panel(self, x, y, width, height):
        pygame.draw.rect(
            self.screen,
            self.PANEL,
            (x, y, width, height),
            border_radius=14,
        )

    def draw_bar(self, label, value, x, y, width, color):
        value = max(0.0, min(1.0, float(value)))

        self.text(label, x, y, self.body_font)

        self.text(
            f"{value * 100:.1f}%",
            x + width - 55,
            y,
            self.small_font,
            self.MUTED,
        )

        pygame.draw.rect(
            self.screen,
            self.PANEL_LIGHT,
            (x, y + 27, width, 11),
            border_radius=6,
        )

        pygame.draw.rect(
            self.screen,
            color,
            (x, y + 27, int(width * value), 11),
            border_radius=6,
        )

    def draw(
        self,
        env,
        info=None,
        episode=1,
        episode_reward=0.0,
        paused=False,
        baseline=True,
    ):
        self.screen.fill(self.BACKGROUND)

        state = env.state

        # Header
        self.text(
            "SmartLearn Ghana",
            30,
            25,
            self.title_font,
        )

        self.text(
            "Adaptive English Learning Simulation",
            32,
            65,
            self.body_font,
            self.MUTED,
        )

        status = "PAUSED" if paused else "RUNNING"

        status_color = self.ORANGE if paused else self.GREEN

        pygame.draw.rect(
            self.screen,
            status_color,
            (925, 35, 135, 35),
            border_radius=10,
        )

        self.text(
            status,
            960,
            44,
            self.small_font,
            self.BACKGROUND,
        )

        # Left panel: learner skills
        self.panel(30, 115, 650, 485)

        self.text(
            "LEARNER PROGRESS",
            55,
            140,
            self.heading_font,
        )

        skill_colors = [
            self.BLUE,
            self.GREEN,
            self.ORANGE,
            (170, 125, 240),
        ]

        for index, (name, value) in enumerate(
            zip(env.SKILLS, state["skills"])
        ):
            y = 195 + index * 85

            self.draw_bar(
                name.replace("_", " ").title(),
                value,
                60,
                y,
                570,
                skill_colors[index],
            )

        # Overall mastery
        mastery = float(state["skills"].mean())

        self.text(
            "Overall mastery",
            60,
            545,
            self.heading_font,
        )

        self.text(
            f"{mastery * 100:.1f}%",
            530,
            545,
            self.heading_font,
            self.GREEN,
        )

        # Right panel: learner wellbeing
        self.panel(705, 115, 365, 300)

        self.text(
            "LEARNER WELLBEING",
            730,
            140,
            self.heading_font,
        )

        self.draw_bar(
            "Confidence",
            state["confidence"],
            730,
            195,
            310,
            self.BLUE,
        )

        self.draw_bar(
            "Engagement",
            state["engagement"],
            730,
            265,
            310,
            self.GREEN,
        )

        self.draw_bar(
            "Fatigue",
            state["fatigue"],
            730,
            335,
            310,
            self.RED,
        )

        # Right panel: session information
        self.panel(705, 435, 365, 165)

        self.text(
            "SESSION INFORMATION",
            730,
            455,
            self.heading_font,
        )

        self.text(
            f"Episode: {episode}",
            730,
            490,
        )

        self.text(
            f"Step: {env.steps}/{env.max_steps}",
            730,
            520,
        )

        self.text(
            f"Reward: {episode_reward:.2f}",
            730,
            550,
            self.body_font,
            self.GREEN,
        )

        # Bottom panel: agent activity
        self.panel(30, 620, 1040, 110)

        self.text(
            "AGENT ACTIVITY",
            55,
            640,
            self.heading_font,
        )

        if baseline:
            self.text(
                "RANDOM BASELINE — NOT A TRAINED AGENT",
                690,
                645,
                self.small_font,
                self.ORANGE,
            )

        if info:
            self.text(
                f"Selected activity: {info['activity']}",
                55,
                675,
                self.body_font,
            )

            result = (
                "Successful activity"
                if info["success"]
                else "Unsuccessful activity"
            )

            result_color = (
                self.GREEN if info["success"] else self.ORANGE
            )

            self.text(
                result,
                500,
                675,
                self.body_font,
                result_color,
            )
        else:
            self.text(
                "Waiting for the first learning activity...",
                55,
                675,
                self.body_font,
                self.MUTED,
            )

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

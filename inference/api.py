
import argparse
import json
import mimetypes
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from environment.custom_env import SmartLearnEnv
from evaluation.agent_loader import TrainedAgent, CHECKPOINTS


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_AGENTS = list(CHECKPOINTS.keys())
SUPPORTED_ALGORITHMS = SUPPORTED_AGENTS + ["RANDOM"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

loaded_agents = {}
sessions = {}


# Ensure JavaScript files are served with the correct MIME type.
mimetypes.add_type("text/javascript", ".js")


# ============================================================
# AGENT LOADING
# ============================================================

def get_agent(algorithm):
    """Load a trained agent once and reuse it."""

    if algorithm not in loaded_agents:
        print(f"\nLoading {algorithm} checkpoint...")

        loaded_agents[algorithm] = TrainedAgent(
            algorithm,
            CHECKPOINTS[algorithm],
        )

        print(f"{algorithm} is ready.\n")

    return loaded_agents[algorithm]


# ============================================================
# LEARNER STATE
# ============================================================

def learner_snapshot(env):
    """Return the current simulated learner state."""

    skills = {
        name: round(float(value), 3)
        for name, value in zip(
            env.SKILLS,
            env.state["skills"],
        )
    }

    return {
        "skills": skills,
        "mean_mastery": round(
            float(np.mean(env.state["skills"])), 3
        ),
        "confidence": round(
            float(env.state["confidence"]), 3
        ),
        "engagement": round(
            float(env.state["engagement"]), 3
        ),
        "fatigue": round(
            float(env.state["fatigue"]), 3
        ),
        "steps": int(env.steps),
        "max_steps": int(env.max_steps),
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_algorithm(value):
    """Validate the selected reinforcement learning agent."""

    if not isinstance(value, str):
        raise ValueError("Algorithm must be a string.")

    algorithm = value.upper()

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm. Choose from: "
            f"{SUPPORTED_ALGORITHMS}"
        )

    return algorithm


# ============================================================
# START SESSION
# ============================================================

def start_session(request_data):
    """Create a new environment episode and initialise its agent."""

    algorithm = validate_algorithm(
        request_data.get("algorithm", "DQN")
    )

    profile = request_data.get("profile", "beginner")

    if not isinstance(profile, str):
        raise ValueError("Learner profile must be a string.")

    seed = request_data.get("seed", 42)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")

    env = SmartLearnEnv(max_steps=20)

    observation, info = env.reset(
        seed=seed,
        options={"profile": profile},
    )

    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "env": env,
        "observation": observation,
        "algorithm": algorithm,
        "profile": profile,
        "episode_reward": 0.0,
        "done": False,
        "rng": np.random.default_rng(seed),
    }

    print("\n" + "=" * 55)
    print("NEW SMARTLEARN SESSION")
    print("=" * 55)
    print(f"Session: {session_id}")
    print(f"Agent:   {algorithm}")
    print(f"Profile: {profile}")
    print("=" * 55 + "\n")

    return {
        "success": True,
        "session_id": session_id,
        "algorithm": algorithm,
        "profile": profile,
        "observation": observation.tolist(),
        "learner": learner_snapshot(env),
    }


# ============================================================
# RESET SESSION
# ============================================================

def reset_session(request_data):
    """Reset an existing episode while retaining its selected agent."""

    session_id = request_data.get("session_id")

    if session_id not in sessions:
        raise ValueError(
            "Session not found. Start a new session."
        )

    session = sessions[session_id]

    seed = request_data.get("seed", 42)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")

    observation, info = session["env"].reset(
        seed=seed,
        options={"profile": session["profile"]},
    )

    session["observation"] = observation
    session["episode_reward"] = 0.0
    session["done"] = False
    session["rng"] = np.random.default_rng(seed)

    print("\n" + "-" * 55)
    print(f"SESSION RESET: {session_id[:8]}")
    print(f"Agent: {session['algorithm']}")
    print(f"Profile: {session['profile']}")
    print("-" * 55 + "\n")

    return {
        "success": True,
        "session_id": session_id,
        "observation": observation.tolist(),
        "learner": learner_snapshot(session["env"]),
    }


# ============================================================
# EXECUTE ONE LEARNING STEP
# ============================================================

def step_session(request_data):
    """Ask the agent to select an action and execute it."""

    session_id = request_data.get("session_id")

    if session_id not in sessions:
        raise ValueError(
            "Session not found. Start a new session."
        )

    session = sessions[session_id]

    if session["done"]:
        raise ValueError(
            "This episode is complete. Please restart it."
        )

    env = session["env"]
    algorithm = session["algorithm"]

    previous_state = learner_snapshot(env)
    observation = session["observation"]

    # Select an action using the chosen agent.
    if algorithm == "RANDOM":
        action = int(
            session["rng"].integers(
                0,
                env.action_space.n,
            )
        )
    else:
        agent = get_agent(algorithm)
        action = int(agent.select_action(observation))

    # Execute the selected action in the actual Gymnasium environment.
    (
        next_observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    done = bool(terminated or truncated)

    # Update session information.
    session["observation"] = next_observation
    session["episode_reward"] += float(reward)
    session["done"] = done

    learner = learner_snapshot(env)

    result = {
        "success": True,
        "session_id": session_id,
        "algorithm": algorithm,
        "action_index": action,
        "activity": info["activity"],
        "target_skill": info["target_skill"],
        "learner_success": bool(info["success"]),
        "success_probability": float(
            info["success_probability"]
        ),
        "reward": float(reward),
        "episode_reward": round(
            session["episode_reward"], 4
        ),
        "step": int(info["steps"]),
        "done": done,
        "termination_reason": info["termination_reason"],
        "previous_state": previous_state,
        "learner": learner,
        "observation": next_observation.tolist(),
    }

    # Detailed terminal output.
    print("\n" + "-" * 65)
    print(f"AGENT:       {algorithm}")
    print(f"EPISODE:     {info['steps']}/{env.max_steps}")
    print(f"ACTION:      {info['activity']}")
    print(f"TARGET:      {info['target_skill']}")
    print(f"SUCCESS:     {info['success']}")
    print(
        f"PROBABILITY: "
        f"{info['success_probability']:.3f}"
    )
    print(f"STEP REWARD: {reward:.3f}")
    print(
        f"TOTAL REWARD: "
        f"{session['episode_reward']:.3f}"
    )
    print(
        f"MEAN MASTERY: "
        f"{learner['mean_mastery']:.3f}"
    )
    print(f"CONFIDENCE:  {learner['confidence']:.3f}")
    print(f"ENGAGEMENT:  {learner['engagement']:.3f}")
    print(f"FATIGUE:     {learner['fatigue']:.3f}")
    print("-" * 65)

    if done:
        print("\nEPISODE COMPLETED")
        print(f"Reason: {info['termination_reason']}")
        print(
            f"Final reward: "
            f"{session['episode_reward']:.3f}"
        )
        print("=" * 65 + "\n")

    return result


# ============================================================
# HTTP SERVER
# ============================================================

class SmartLearnAPIHandler(BaseHTTPRequestHandler):
    """Serve the SmartLearn frontend and simulation API."""

    def send_json(self, status_code, response):
        response_bytes = json.dumps(response).encode("utf-8")

        self.send_response(status_code)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(response_bytes)),
        )
        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )
        self.end_headers()

        self.wfile.write(response_bytes)

    def read_json(self):
        """Read and validate a JSON request."""

        content_length = int(
            self.headers.get("Content-Length", 0)
        )

        if content_length <= 0 or content_length > 10000:
            raise ValueError("Invalid request size.")

        request_body = self.rfile.read(content_length)

        try:
            request_data = json.loads(
                request_body.decode("utf-8")
            )
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON request.")

        if not isinstance(request_data, dict):
            raise ValueError(
                "The request must be a JSON object."
            )

        return request_data

    def do_OPTIONS(self):
        self.send_json(
            200,
            {"success": True},
        )

    # --------------------------------------------------------
    # GET: HEALTH CHECK AND FRONTEND FILES
    # --------------------------------------------------------

    def do_GET(self):

        requested_path = unquote(
            self.path.split("?")[0]
        )

        # API health check.
        if requested_path == "/health":
            self.send_json(
                200,
                {
                    "status": "running",
                    "service": "SmartLearn Ghana",
                    "supported_agents": SUPPORTED_ALGORITHMS,
                    "active_sessions": len(sessions),
                },
            )
            return

        # Homepage.
        if requested_path == "/":
            requested_path = "/index.html"

        # Avoid unnecessary favicon errors.
        if requested_path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        # Resolve requested file inside frontend directory.
        file_path = (
            FRONTEND_DIR / requested_path.lstrip("/")
        ).resolve()

        # Security: prevent access outside frontend directory.
        if (
            file_path != FRONTEND_DIR
            and FRONTEND_DIR not in file_path.parents
        ):
            self.send_error(403, "Access denied")
            return

        if not file_path.is_file():
            self.send_error(
                404,
                "Frontend file not found",
            )
            return

        content_type, _ = mimetypes.guess_type(
            str(file_path)
        )

        if content_type is None:
            content_type = "application/octet-stream"

        content = file_path.read_bytes()

        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"{content_type}; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(content)),
        )
        self.send_header(
            "Cache-Control",
            "no-cache",
        )
        self.end_headers()

        self.wfile.write(content)

    # --------------------------------------------------------
    # POST: SIMULATION ENDPOINTS
    # --------------------------------------------------------

    def do_POST(self):

        try:
            request_data = self.read_json()

            if self.path == "/session/start":
                result = start_session(request_data)

            elif self.path == "/session/step":
                result = step_session(request_data)

            elif self.path == "/session/reset":
                result = reset_session(request_data)

            else:
                self.send_json(
                    404,
                    {
                        "success": False,
                        "error": "Endpoint not found.",
                    },
                )
                return

            self.send_json(200, result)

        except ValueError as error:
            self.send_json(
                400,
                {
                    "success": False,
                    "error": str(error),
                },
            )

        except Exception as error:
            print("\nSIMULATION API ERROR")
            print(f"Details: {error}\n")

            self.send_json(
                500,
                {
                    "success": False,
                    "error": (
                        "The simulation request "
                        "could not be completed."
                    ),
                },
            )

    def log_message(self, format, *args):
        print(f"HTTP: {format % args}")


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="SmartLearn Ghana Application"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host address.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port.",
    )

    args = parser.parse_args()

    if not FRONTEND_DIR.is_dir():
        raise FileNotFoundError(
            f"Frontend directory not found: {FRONTEND_DIR}"
        )

    if not (FRONTEND_DIR / "index.html").is_file():
        raise FileNotFoundError(
            f"Frontend homepage not found: "
            f"{FRONTEND_DIR / 'index.html'}"
        )

    server = HTTPServer(
        (args.host, args.port),
        SmartLearnAPIHandler,
    )

    print("\n" + "=" * 52)
    print("SMARTLEARN GHANA")
    print("=" * 52)
    print(
        f"Application: http://{args.host}:{args.port}"
    )
    print(
        f"API health:  http://{args.host}:{args.port}/health"
    )
    print("Session start: POST /session/start")
    print("Session step:  POST /session/step")
    print("Session reset: POST /session/reset")
    print("=" * 52)
    print("Press CTRL+C to stop the server.\n")

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nShutting down SmartLearn Ghana...")

    finally:
        for session in sessions.values():
            session["env"].close()

        server.server_close()


if __name__ == "__main__":
    main()

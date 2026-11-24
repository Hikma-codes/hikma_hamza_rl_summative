
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

from evaluation.agent_loader import TrainedAgent, CHECKPOINTS


ACTION_NAMES = [
    "Vocabulary introduction",
    "Listening practice",
    "Local-language support scaffold",
    "Word matching",
    "Reading comprehension",
    "Sentence building",
    "Revision of weakest skill",
]

OBSERVATION_SIZE = 11
SUPPORTED_AGENTS = list(CHECKPOINTS.keys())

# Cache loaded models so they are not reloaded for every request.
loaded_agents = {}


def get_agent(algorithm):
    """Load an agent once and reuse it for later requests."""

    if algorithm not in loaded_agents:
        print(f"Loading {algorithm} checkpoint...")

        loaded_agents[algorithm] = TrainedAgent(
            algorithm,
            CHECKPOINTS[algorithm],
        )

        print(f"{algorithm} is ready.")

    return loaded_agents[algorithm]


def predict(request_data):
    """Validate a JSON request and return the agent's recommendation."""

    if not isinstance(request_data, dict):
        raise ValueError("The request must be a JSON object.")

    algorithm = request_data.get("algorithm", "DQN")

    if not isinstance(algorithm, str):
        raise ValueError("Algorithm must be a string.")

    algorithm = algorithm.upper()

    if algorithm not in SUPPORTED_AGENTS:
        raise ValueError(
            f"Unsupported algorithm. Choose from: {SUPPORTED_AGENTS}"
        )

    observation = request_data.get("observation")

    if not isinstance(observation, list):
        raise ValueError("Observation must be a list of 11 numbers.")

    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(
            f"Expected {OBSERVATION_SIZE} observation values, "
            f"received {len(observation)}."
        )

    validated_observation = []

    for index, value in enumerate(observation):

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Observation value {index + 1} must be a number."
            )

        value = float(value)

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Observation value {index + 1} must be between 0 and 1."
            )

        validated_observation.append(value)

    agent = get_agent(algorithm)

    action = agent.select_action(validated_observation)

    return {
        "success": True,
        "algorithm": algorithm,
        "action_index": action,
        "recommended_activity": ACTION_NAMES[action],
    }


class SmartLearnAPIHandler(BaseHTTPRequestHandler):
    """Handle SmartLearn JSON API requests."""

    def send_json(self, status_code, response):
        response_bytes = json.dumps(response).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
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

    def do_OPTIONS(self):
        self.send_json(200, {"success": True})

    def do_GET(self):

        if self.path == "/health":
            self.send_json(
                200,
                {
                    "status": "running",
                    "service": "SmartLearn Ghana Inference API",
                    "supported_agents": SUPPORTED_AGENTS,
                },
            )

        else:
            self.send_json(
                404,
                {"success": False, "error": "Endpoint not found."},
            )

    def do_POST(self):

        if self.path != "/predict":
            self.send_json(
                404,
                {"success": False, "error": "Endpoint not found."},
            )
            return

        try:
            content_length = int(
                self.headers.get("Content-Length", 0)
            )

            if content_length <= 0 or content_length > 10000:
                raise ValueError("Invalid request size.")

            request_body = self.rfile.read(content_length)

            request_data = json.loads(
                request_body.decode("utf-8")
            )

            result = predict(request_data)

            self.send_json(200, result)

        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(
                400,
                {
                    "success": False,
                    "error": str(error),
                },
            )

        except Exception as error:
            print(f"Inference error: {error}")

            self.send_json(
                500,
                {
                    "success": False,
                    "error": "The inference request could not be completed.",
                },
            )

    def log_message(self, format, *args):
        print(f"API: {format % args}")


def main():
    parser = argparse.ArgumentParser(
        description="SmartLearn Ghana JSON Inference API"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API host address.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port.",
    )

    args = parser.parse_args()

    server = HTTPServer(
        (args.host, args.port),
        SmartLearnAPIHandler,
    )

    print("\nSmartLearn Ghana Inference API")
    print(f"Server running at http://{args.host}:{args.port}")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print(f"Prediction endpoint: http://{args.host}:{args.port}/predict")
    print("Press CTRL+C to stop the server.\n")

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nShutting down the API...")

    finally:
        server.server_close()


if __name__ == "__main__":
    main()


const API_BASE = "";

let sessionId = null;
let currentAlgorithm = "DQN";
let currentProfile = "beginner";
let sessionDone = false;
let sessionReward = 0;
let history = [];
let lastObservation = null;
let toastTimeout = null;

const activityDetails = {
    0: {
        symbol: "A",
        title: "Vocabulary introduction",
        description: "Discover new English words and build your vocabulary.",
        message: "Let's discover some new words!"
    },
    1: {
        symbol: "L",
        title: "Listening practice",
        description: "Practise listening and understanding spoken English.",
        message: "Listen carefully and learn!"
    },
    2: {
        symbol: "Aa",
        title: "Local-language support",
        description: "Use language support to make English learning more accessible.",
        message: "Every language can help us learn!"
    },
    3: {
        symbol: "W",
        title: "Word matching",
        description: "Connect words with their meanings and practise recognition.",
        message: "Can you match the words?"
    },
    4: {
        symbol: "R",
        title: "Reading comprehension",
        description: "Practise understanding written English and its meaning.",
        message: "Let's read and understand!"
    },
    5: {
        symbol: "Aa",
        title: "Sentence building",
        description: "Put words together to communicate ideas clearly.",
        message: "Let's build a sentence!"
    },
    6: {
        symbol: "R",
        title: "Revision of weakest skill",
        description: "Revisit the skill that currently needs the most support.",
        message: "Practice makes progress!"
    }
};

const observationLabels = [
    "Vocabulary mastery",
    "Listening mastery",
    "Comprehension mastery",
    "Sentence building",
    "Confidence",
    "Engagement",
    "Fatigue",
    "Difficulty",
    "Previous success",
    "Remaining time",
    "Recent mistakes"
];

const skillObservationIndices = {
    vocabulary: 0,
    listening: 1,
    comprehension: 2,
    sentence_building: 3
};

const $ = (id) => document.getElementById(id);

async function apiRequest(endpoint, body = null) {
    const options = {
        method: body === null ? "GET" : "POST",
        headers: {}
    };

    if (body !== null) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
    }

    const response = await fetch(`${API_BASE}${endpoint}`, options);

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error("The server returned an unreadable response.");
    }

    if (!response.ok || data.success === false) {
        throw new Error(data.error || "The request could not be completed.");
    }

    return data;
}

function showToast(message) {
    const toast = $("toast");

    toast.textContent = message;
    toast.classList.add("show");

    clearTimeout(toastTimeout);

    toastTimeout = setTimeout(() => {
        toast.classList.remove("show");
    }, 3500);
}

function setConnection(online) {
    $("connection-dot").className = online
        ? "status-dot online"
        : "status-dot offline";

    $("connection-label").textContent = online
        ? "System connected"
        : "System offline";
}

async function checkConnection() {
    try {
        await apiRequest("/health");
        setConnection(true);
    } catch (error) {
        setConnection(false);
        console.error("Connection check failed:", error);
    }
}

function switchView(viewId) {
    document.querySelectorAll(".page-view").forEach((view) => {
        view.classList.toggle("active-view", view.id === viewId);
    });

    document.querySelectorAll(".nav-link").forEach((button) => {
        button.classList.toggle(
            "active",
            button.dataset.view === viewId
        );
    });
}

function updateProgressBar(barId, value) {
    const bar = $(barId);

    if (bar) {
        bar.style.width = `${Math.round(value * 100)}%`;
    }
}

function updatePercentage(labelId, value) {
    const label = $(labelId);

    if (label) {
        label.textContent = `${Math.round(value * 100)}%`;
    }
}

function renderLearner(learner) {
    if (!learner) return;

    const skills = learner.skills;

    const skillMappings = [
        ["vocabulary", "vocabulary-bar", "vocabulary-value"],
        ["listening", "listening-bar", "listening-value"],
        ["comprehension", "comprehension-bar", "comprehension-value"],
        ["sentence_building", "sentence_building-bar", "sentence_building-value"]
    ];

    skillMappings.forEach(([key, barId, valueId]) => {
        const value = skills[key] ?? 0;
        updateProgressBar(barId, value);
        updatePercentage(valueId, value);
    });

    updatePercentage("confidence-value", learner.confidence);
    updatePercentage("engagement-value", learner.engagement);
    updatePercentage("fatigue-value", learner.fatigue);

    updateProgressBar("confidence-bar", learner.confidence);
    updateProgressBar("engagement-bar", learner.engagement);
    updateProgressBar("fatigue-bar", learner.fatigue);

    $("mastery-total").textContent =
        `${Math.round(learner.mean_mastery * 100)}% overall mastery`;

    $("step-count").textContent = learner.steps;
    $("journey-count").textContent =
        `${learner.steps} / ${learner.max_steps}`;

    updateProgressBar(
        "journey-bar",
        learner.steps / learner.max_steps
    );

    $("reward-total").textContent = sessionReward.toFixed(2);
}

function renderObservation(observation) {
    if (!Array.isArray(observation)) return;

    lastObservation = observation;

    const grid = $("observation-grid");
    grid.innerHTML = "";

    observation.forEach((value, index) => {
        const item = document.createElement("div");
        item.className = "observation-item";

        const label = document.createElement("span");
        label.textContent = observationLabels[index] || `Value ${index + 1}`;

        const number = document.createElement("strong");
        number.textContent = Number(value).toFixed(3);

        item.append(label, number);
        grid.appendChild(item);
    });
}

/*
 * Creates an observation-based explanation of the agent's decision.
 *
 * The agent outputs an action, not a written explanation.
 * This function summarises the learner-state information that
 * was available to the agent when it selected that action.
 */
function buildDecisionExplanation(result) {
    const observation = result.observation;

    if (!Array.isArray(observation) || observation.length < 11) {
        return `The ${result.algorithm} agent selected ${activityDetails[result.action_index]?.title || "an activity"} using the learner's current observation. The environment then executed the action and returned the outcome.`;
    }

    const targetSkill = String(result.target_skill || "")
        .toLowerCase()
        .replaceAll(" ", "_");

    const targetIndex = skillObservationIndices[targetSkill];

    const targetMastery = targetIndex !== undefined
        ? Number(observation[targetIndex])
        : null;

    const skillLevels = Object.entries(skillObservationIndices)
        .map(([skill, index]) => ({
            skill,
            value: Number(observation[index])
        }))
        .sort((a, b) => a.value - b.value);

    const lowestSkill = skillLevels[0];

    const confidence = Number(observation[4]);
    const engagement = Number(observation[5]);
    const fatigue = Number(observation[6]);
    const difficulty = Number(observation[7]);

    const targetLabel = targetSkill
        .replaceAll("_", " ");

    const activity = activityDetails[result.action_index]?.title
        || "an activity";

    const targetContext = targetMastery !== null
        ? `The current ${targetLabel} mastery level is ${Math.round(targetMastery * 100)}%.`
        : `The selected activity is associated with ${targetLabel}.`;

    return [
        `The ${result.algorithm} agent selected ${activity} after receiving the simulated learner's 11-value observation.`,
        targetContext,
        `Among the four tracked skills, ${lowestSkill.skill.replaceAll("_", " ")} currently has the lowest mastery level (${Math.round(lowestSkill.value * 100)}%).`,
        `Other information available to the agent included confidence (${Math.round(confidence * 100)}%), engagement (${Math.round(engagement * 100)}%), fatigue (${Math.round(fatigue * 100)}%) and difficulty (${Math.round(difficulty * 100)}%).`,
        `This is an observation-based summary of the information available when the action was selected. It does not claim to reveal the neural network's exact internal reasoning.`
    ].join(" ");
}

function renderStateChanges(previousState, currentState) {
    const container = $("state-changes");
    container.innerHTML = "";

    const entries = [
        ["Vocabulary", "vocabulary"],
        ["Listening", "listening"],
        ["Comprehension", "comprehension"],
        ["Sentence building", "sentence_building"],
        ["Confidence", "confidence"],
        ["Engagement", "engagement"],
        ["Fatigue", "fatigue"]
    ];

    entries.forEach(([label, key]) => {
        const previous = key in previousState.skills
            ? previousState.skills[key]
            : previousState[key];

        const current = key in currentState.skills
            ? currentState.skills[key]
            : currentState[key];

        const difference = current - previous;

        const row = document.createElement("div");
        row.className = "state-row";

        const name = document.createElement("span");
        name.textContent = label;

        const bar = document.createElement("div");
        bar.className = "state-bar";

        const fill = document.createElement("span");
        fill.style.width = `${Math.max(0, Math.min(100, current * 100))}%`;
        bar.appendChild(fill);

        const amount = document.createElement("strong");
        amount.textContent =
            `${Math.round(current * 100)}% (${difference >= 0 ? "+" : ""}${Math.round(difference * 100)}%)`;

        row.append(name, bar, amount);
        container.appendChild(row);
    });
}

function renderDecision(result) {
    const details = activityDetails[result.action_index];

    $("activity-symbol").textContent = details.symbol;
    $("activity-title").textContent = details.title;
    $("activity-description").textContent = details.description;
    $("activity-number").textContent = `STEP ${result.step}`;
    $("target-skill").textContent =
        `Focus: ${result.target_skill.replaceAll("_", " ")}`;
    $("activity-difficulty").textContent =
        `Success chance: ${Math.round(result.success_probability * 100)}%`;

    $("scene-message").textContent = details.message;

    $("buddy-message").textContent = result.learner_success
        ? "Wonderful progress! Every step helps you grow."
        : "That's okay! Learning takes practice. Keep going.";

    $("step-button").textContent = result.done
        ? "Session completed"
        : "Try the next activity →";

    $("step-button").disabled = result.done;

    $("lab-algorithm").textContent = result.algorithm;
    $("lab-action").textContent = details.title;
    $("lab-reward").textContent = Number(result.reward).toFixed(3);
    $("lab-probability").textContent =
        `${Math.round(result.success_probability * 100)}%`;

    $("decision-activity").textContent = details.title;

    // New observation-based decision explanation.
    $("decision-explanation").textContent =
        buildDecisionExplanation(result);

    $("decision-index").textContent = result.action_index;
    $("decision-target").textContent =
        result.target_skill.replaceAll("_", " ");

    $("decision-outcome").textContent = result.learner_success
        ? "Success"
        : "Needs more practice";

    $("decision-outcome").style.color = result.learner_success
        ? "#289c75"
        : "#d77c57";

    renderStateChanges(result.previous_state, result.learner);
    renderObservation(result.observation);

    addHistory(result);
}

function addHistory(result) {
    history.unshift(result);

    $("history-count").textContent =
        `${history.length} actions`;

    const list = $("history-list");
    list.innerHTML = "";

    history.slice(0, 20).forEach((item) => {
        const details = activityDetails[item.action_index];

        const row = document.createElement("div");
        row.className = "history-item";

        const index = document.createElement("span");
        index.className = "history-index";
        index.textContent = item.step;

        const description = document.createElement("div");

        const title = document.createElement("strong");
        title.textContent = details.title;

        const subtitle = document.createElement("small");
        subtitle.textContent =
            `${item.algorithm} · ${item.learner_success ? "Successful outcome" : "Unsuccessful outcome"}`;

        description.append(title, subtitle);

        const reward = document.createElement("span");
        reward.className = "history-reward";
        reward.textContent = Number(item.reward).toFixed(3);

        row.append(index, description, reward);
        list.appendChild(row);
    });
}

function setSessionStatus(status, text) {
    const pill = $("session-status");
    pill.className = `session-pill ${status}`;
    pill.textContent = text;
}

async function startSession() {
    const button = $("start-button");

    button.disabled = true;
    button.textContent = "Starting your adventure...";

    try {
        currentAlgorithm = $("algorithm-select").value;
        currentProfile = $("profile-select").value;

        const result = await apiRequest("/session/start", {
            algorithm: currentAlgorithm,
            profile: currentProfile,
            seed: 42
        });

        sessionId = result.session_id;
        sessionDone = false;
        sessionReward = 0;
        history = [];

        $("history-list").innerHTML =
            '<p class="empty-state">Your agent is ready. Complete an activity to begin the timeline.</p>';

        $("history-count").textContent = "0 actions";
        $("lab-algorithm").textContent = currentAlgorithm;
        $("lab-action").textContent = "Waiting";
        $("lab-reward").textContent = "0.00";
        $("lab-probability").textContent = "—";
        $("decision-activity").textContent = "Ready for its first decision";
        $("decision-explanation").textContent =
            "The trained agent will receive the simulated learner's current state and select an activity. After the first action, this panel will explain the relevant observation values.";

        $("decision-index").textContent = "—";
        $("decision-target").textContent = "—";
        $("decision-outcome").textContent = "—";

        $("state-changes").innerHTML =
            '<p class="empty-state">State comparisons will appear after the first agent action.</p>';

        renderLearner(result.learner);
        renderObservation(result.observation);

        $("activity-symbol").textContent = "A";
        $("activity-title").textContent = "Your adventure is ready!";
        $("activity-description").textContent =
            "Your learning guide is ready. Start the first simulated activity to see which learning action it selects.";
        $("activity-number").textContent = "READY";
        $("target-skill").textContent = "Waiting for first action";
        $("activity-difficulty").textContent = "Adaptive learning";
        $("step-button").disabled = false;
        $("step-button").textContent = "Start first activity →";

        $("buddy-message").textContent =
            "Your learning adventure has begun. Let's take the first step!";

        $("scene-message").textContent =
            "Your learning adventure has begun!";

        setSessionStatus("running", "Session active");

        button.textContent = "Restart adventure";
        showToast(`${currentAlgorithm} learning session started.`);
        setConnection(true);
    } catch (error) {
        showToast(error.message);
        console.error(error);
    } finally {
        button.disabled = false;

        if (!sessionId) {
            button.textContent = "Start my adventure →";
        }
    }
}

async function performStep() {
    if (!sessionId || sessionDone) return;

    const button = $("step-button");
    button.disabled = true;
    button.textContent = "Learning in progress...";

    try {
        const result = await apiRequest("/session/step", {
            session_id: sessionId
        });

        sessionReward = result.episode_reward;
        sessionDone = result.done;

        renderLearner(result.learner);
        renderDecision(result);

        if (result.done) {
            setSessionStatus("finished", "Session complete");
            $("buddy-message").textContent =
                "You did it! What a wonderful learning adventure.";
            $("scene-message").textContent =
                "Wonderful work! Session complete.";
            showToast(`Session complete. Total reward: ${sessionReward.toFixed(2)}`);
        } else {
            setSessionStatus("running", "Session active");

            showToast(
                result.learner_success
                    ? "Great! The simulated learner made progress."
                    : "The simulated learner needs more practice."
            );
        }

        setConnection(true);
    } catch (error) {
        showToast(error.message);
        console.error(error);
    } finally {
        button.disabled = sessionDone || !sessionId;
    }
}

async function resetSession() {
    const selectedAlgorithm = $("algorithm-select").value;
    const selectedProfile = $("profile-select").value;

    if (
        !sessionId ||
        selectedAlgorithm !== currentAlgorithm ||
        selectedProfile !== currentProfile
    ) {
        sessionId = null;
        await startSession();
        return;
    }

    const button = $("start-button");
    button.disabled = true;
    button.textContent = "Resetting...";

    try {
        const result = await apiRequest("/session/reset", {
            session_id: sessionId,
            seed: 42
        });

        sessionDone = false;
        sessionReward = 0;
        history = [];

        $("history-list").innerHTML =
            '<p class="empty-state">The agent has not made a decision in this new episode yet.</p>';

        $("history-count").textContent = "0 actions";

        $("lab-action").textContent = "Waiting";
        $("lab-reward").textContent = "0.00";
        $("lab-probability").textContent = "—";
        $("decision-activity").textContent = "Ready for its first decision";
        $("decision-explanation").textContent =
            "The agent is ready to receive the new simulated learner state and make another decision.";

        $("decision-index").textContent = "—";
        $("decision-target").textContent = "—";
        $("decision-outcome").textContent = "—";

        $("state-changes").innerHTML =
            '<p class="empty-state">State comparisons will appear after the first agent action.</p>';

        $("activity-title").textContent = "Your adventure is ready!";
        $("activity-description").textContent =
            "Start the first activity to see what your learning guide recommends.";
        $("activity-symbol").textContent = "A";
        $("activity-number").textContent = "READY";
        $("target-skill").textContent = "Waiting for first action";
        $("activity-difficulty").textContent = "Adaptive learning";
        $("step-button").textContent = "Start first activity →";
        $("step-button").disabled = false;

        $("buddy-message").textContent =
            "A fresh start! Let's learn something new.";

        $("scene-message").textContent =
            "A new learning adventure begins!";

        renderLearner(result.learner);
        renderObservation(result.observation);

        setSessionStatus("running", "Session active");
        showToast("Your learning session has been reset.");
    } catch (error) {
        showToast(error.message);
        console.error(error);
    } finally {
        button.disabled = false;
        button.textContent = "Restart adventure";
    }
}

function setupNavigation() {
    document.querySelectorAll(".nav-link").forEach((button) => {
        button.addEventListener("click", () => {
            switchView(button.dataset.view);
        });
    });
}

function setupControls() {
    $("start-button").addEventListener("click", resetSession);
    $("step-button").addEventListener("click", performStep);

    $("algorithm-select").addEventListener("change", () => {
        if (sessionId) {
            showToast(
                "To use a different agent, restart the adventure."
            );
        }
    });

    $("profile-select").addEventListener("change", () => {
        if (sessionId) {
            showToast(
                "Restart the adventure to apply a different learner profile."
            );
        }
    });
}

function initialise() {
    setupNavigation();
    setupControls();
    checkConnection();

    setInterval(checkConnection, 15000);

    console.log("SmartLearn Ghana frontend initialised.");
}

document.addEventListener("DOMContentLoaded", initialise);

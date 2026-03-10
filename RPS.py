import math

ROCK = "R"
PAPER = "P"
SCISSORS = "S"

MOVES = (ROCK, PAPER, SCISSORS)

# "beats[x]" is the move that beats x.
beats = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}

data = {"history": []}  # kept for main.py convenience


def calculate_tuple_freq_distribution(freq_history: list[str]):
    """
    Utility kept for your local exploration in main.py.
    Returns (pair_counts, row_normalized_pair_probs).
    """
    pairs = [(a, b) for a in MOVES for b in MOVES]
    pair_counts = {p: 0 for p in pairs}

    for i in range(1, len(freq_history) - 1):
        a, b = freq_history[i], freq_history[i + 1]
        if a in MOVES and b in MOVES:
            pair_counts[(a, b)] += 1

    row_probs = {p: 0.0 for p in pairs}
    for a in MOVES:
        row_total = sum(pair_counts[(a, b)] for b in MOVES)
        for b in MOVES:
            row_probs[(a, b)] = (pair_counts[(a, b)] / row_total) if row_total else 0.0

    return pair_counts, row_probs


def _outcome(my_play: str, opp_play: str) -> int:
    if my_play == opp_play:
        return 0
    if beats[opp_play] == my_play:
        return 1
    return -1


def _predict_mrugesh_and_update(my_prev: str, mrugesh_history: list[str]) -> str:
    # mrugesh() appends the previous opponent play (our previous play) each round
    mrugesh_history.append(my_prev)
    last_ten = mrugesh_history[-10:]
    most_frequent = max(set(last_ten), key=last_ten.count)
    if most_frequent == "":
        most_frequent = SCISSORS
    return beats[most_frequent]


def _predict_abbey_and_update(
    my_prev: str,
    abbey_history: list[str],
    play_order: dict[str, int],
) -> str:
    # abbey() replaces empty previous play with 'R'
    if not my_prev:
        my_prev = ROCK
    abbey_history.append(my_prev)

    last_two = "".join(abbey_history[-2:])
    if len(last_two) == 2:
        play_order[last_two] += 1

    potential_plays = [my_prev + ROCK, my_prev + PAPER, my_prev + SCISSORS]
    sub_order = {k: play_order[k] for k in potential_plays}
    prediction = max(sub_order, key=sub_order.get)[-1:]
    return beats[prediction]


def _reset_state(state: dict) -> None:
    state.clear()
    state.update(
        {
            "_init": True,
            "my_history": [],
            "opp_history": [],
            "scores": {"quincy": 0, "abbey": 0, "kris": 0, "mrugesh": 0},
            "pred_last": {"quincy": None, "abbey": None, "kris": None, "mrugesh": None},
            # model states
            "quincy_counter": 0,
            "quincy_choices": [ROCK, ROCK, PAPER, PAPER, SCISSORS],
            "mrugesh_history": [],
            "abbey_history": [],
            "abbey_play_order": {a + b: 0 for a in MOVES for b in MOVES},
        }
    )


def player(prev_play: str, state: dict = {"_init": False}) -> str:
    """
    Adaptive player:
    - Simulates all four known bots to predict their next move
    - Scores which simulated bot best matches observed opponent behavior
    - Plays the counter to the highest-confidence prediction
    """
    if (not state.get("_init")) or prev_play == "":
        _reset_state(state)

    if prev_play:
        state["opp_history"].append(prev_play)
        data["history"] = state["opp_history"]

        # Update model scores based on how well each model predicted last round.
        for bot, pred in state["pred_last"].items():
            if pred is None:
                continue
            state["scores"][bot] += 2 if pred == prev_play else -1

    my_prev = state["my_history"][-1] if state["my_history"] else ""

    # Predict opponent's play for the *current* round using each model.
    preds: dict[str, str] = {}

    state["quincy_counter"] += 1
    preds["quincy"] = state["quincy_choices"][state["quincy_counter"] % len(state["quincy_choices"])]

    preds["kris"] = beats[(my_prev or ROCK)]

    preds["mrugesh"] = _predict_mrugesh_and_update(my_prev, state["mrugesh_history"])
    preds["abbey"] = _predict_abbey_and_update(my_prev, state["abbey_history"], state["abbey_play_order"])

    # Choose our play by maximizing expected outcome under weighted model belief.
    # (This hedges early when we haven't identified the opponent yet.)
    weights = {b: math.exp(state["scores"][b] / 3.0) for b in state["scores"]}

    def expected_value(my_play: str) -> float:
        return sum(weights[b] * _outcome(my_play, preds[b]) for b in preds)

    my_play = max(MOVES, key=expected_value)

    state["my_history"].append(my_play)
    state["pred_last"] = preds
    return my_play
import numpy as np

from trajmem_ot.pair_mining import ContextRecord, mine_disjoint_context_pairs


def _record(index, episode, prompt="p", subgoal="s", value=0.0):
    return ContextRecord(
        index=index,
        episode=episode,
        prompt=prompt,
        subgoal=subgoal,
        front=np.full((2, 2, 3), value, dtype=np.float32),
        wrist=np.full((2, 2, 3), value, dtype=np.float32),
        state=np.asarray([value], dtype=np.float32),
    )


def test_miner_selects_lowest_cost_episode_disjoint_pairs():
    records = [
        _record(10, 1, value=0.00),
        _record(20, 2, value=0.01),
        _record(30, 3, value=0.20),
        _record(40, 4, value=0.21),
    ]

    pairs = mine_disjoint_context_pairs(
        records,
        max_front_mae=0.05,
        max_wrist_mae=0.05,
        max_state_l2=0.05,
    )

    assert {tuple((row["a"], row["b"])) for row in pairs} == {(10, 20), (30, 40)}
    assert len({episode for row in pairs for episode in row["episodes"]}) == 4


def test_miner_requires_same_prompt_and_subgoal():
    records = [
        _record(10, 1, prompt="p1"),
        _record(20, 2, prompt="p2"),
        _record(30, 3, prompt="p1", subgoal="other"),
    ]

    assert mine_disjoint_context_pairs(records) == []

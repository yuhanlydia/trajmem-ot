import torch

from trajmem_ot.core import MemoryEditConfig, optimize_memory_ot, sinkhorn
from trajmem_ot.adapters import extract_robomme_history, replace_robomme_history
from trajmem_ot.robomme_branch import observation_fingerprint
from trajmem_ot.evaluation import select_trust_radius, summarize_memory_line_search, summarize_paired_return


def test_sinkhorn_marginals():
    p = torch.tensor([0.25, 0.75])
    q = torch.tensor([0.6, 0.4])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    coupling = sinkhorn(p, q, cost, epsilon=0.2, iters=200)
    torch.testing.assert_close(coupling.sum(1), p, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(coupling.sum(0), q, atol=1e-5, rtol=1e-5)


def test_memory_update_is_bounded_and_history_shaped():
    memory = torch.randn(6, 2, 4)
    noises = torch.randn(4, 3, 2)
    weight = torch.randn(24, 2)

    def policy(mem, eps):
        mean = mem.mean(1).flatten() @ weight
        return eps + mean

    actions = policy(memory, noises)
    returns = -actions.square().mean((1, 2))
    cfg = MemoryEditConfig(trust_radius=0.01, rank=2, temporal_topk=2, cg_iters=8)
    result = optimize_memory_ot(memory, noises, returns, policy, cfg)
    assert result.delta.shape == memory.shape
    assert result.delta.norm() <= 0.01001 * memory.norm()
    assert (result.temporal_saliency > 0).sum() <= 2


def test_robomme_adapter_cannot_modify_current_inputs():
    batch = {
        "images": torch.randn(1, 3, 224, 224),
        "state": torch.randn(1, 14),
        "tokenized_prompt": torch.ones(1, 16, dtype=torch.long),
        "recur_image_emb": torch.randn(1, 8, 2, 64, 32),
        "recur_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    history = extract_robomme_history(batch)
    changed = replace_robomme_history(batch, history, history.values + 0.01)
    assert changed["images"] is batch["images"]
    assert changed["state"] is batch["state"]
    assert changed["tokenized_prompt"] is batch["tokenized_prompt"]
    assert not torch.equal(changed["recur_image_emb"], batch["recur_image_emb"])


def test_observation_fingerprint_is_content_sensitive():
    observation = {
        "front_rgb_list": [torch.zeros(2, 2, 3, dtype=torch.uint8).numpy()],
        "wrist_rgb_list": [torch.zeros(2, 2, 3, dtype=torch.uint8).numpy()],
        "joint_state_list": [torch.zeros(7).numpy()],
        "gripper_state_list": [torch.ones(1).numpy()],
    }
    first = observation_fingerprint(observation)
    observation["front_rgb_list"][0][0, 0, 0] = 1
    assert observation_fingerprint(observation) != first


def test_e7_selection_does_not_promote_failed_descriptive_radius():
    rows = [
        {"q0": 0.0, "sweep": [
            {"rho": 0.0, "q_plus": 0.0, "q_minus": 0.0, "q_random": []},
            {"rho": 0.001, "q_plus": 0.1, "q_minus": -0.1, "q_random": [0.0]},
        ]},
        {"q0": 0.0, "sweep": [
            {"rho": 0.0, "q_plus": 0.0, "q_minus": 0.0, "q_random": []},
            {"rho": 0.001, "q_plus": -0.1, "q_minus": 0.1, "q_random": [0.0]},
        ]},
    ]
    summary = summarize_memory_line_search(rows)
    assert summary[0]["p_positive_central_slope"] == 0.5
    assert select_trust_radius(summary) is None


def test_e7_selects_minimum_passing_low_energy_radius():
    summaries = [
        {"rho": 0.00003125, "p_positive_central_slope": 0.64, "median_delta_q": 1e-6},
        {"rho": 0.000125, "p_positive_central_slope": 0.72, "median_delta_q": 2e-6},
        {"rho": 0.00025, "p_positive_central_slope": 0.75, "median_delta_q": 5e-6},
    ]
    assert select_trust_radius(summaries) == 0.000125


def test_e8_summary_uses_matched_branch_and_noise_pairs():
    rows = []
    for seed, base_success, plus_success in [(1, False, True), (2, True, True)]:
        for condition, success in [("M", base_success), ("M_plus", plus_success)]:
            rows.append({"task": "T", "episode": 0, "branch_step": 30, "noise_seed": seed,
                         "condition": condition, "success": success,
                         "return": float(success), "steps": 100})
    report = summarize_paired_return(rows)
    assert report["n_pairs"] == 2
    assert report["by_condition"]["M_plus"]["success_rate"] == 1.0
    assert report["paired_vs_baseline"]["M_plus"]["success_gains"] == 1

"""Check paired evaluation artifacts before accepting a run's results."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(directory, expected_count):
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    cfg = manifest["config"]
    assert cfg["solver"]["n_steps"] == 10
    assert cfg["solver"]["num_samples"] == 300 and cfg["solver"]["topk"] == 30
    assert cfg["solver"]["batch_size"] == 1
    assert cfg["plan_config"]["horizon"] == cfg["plan_config"]["receding_horizon"] == 5
    assert cfg["plan_config"]["action_block"] == 5 and cfg["plan_config"]["history_len"] == 1
    assert cfg["eval"]["eval_budget"] == 50 and cfg["eval"]["goal_offset_steps"] == 25
    assert cfg["eval"]["num_eval"] == expected_count
    assert len(manifest["dataset_rows"]) == len(metrics["episode_successes"]) == expected_count
    assert abs(metrics["success_rate"] - np.mean(metrics["episode_successes"]) * 100) < 1e-8
    traces = np.load(directory / "rollout.npz")
    assert traces["actions"].shape[:2] == (metrics["steps_executed"], expected_count)
    assert np.isfinite(traces["actions"][traces["active"]]).all()
    assert np.array_equal(traces["successes"][-1], metrics["episode_successes"])
    assert np.array_equal(traces["first_success_step"] > 0, metrics["episode_successes"])
    assert len(metrics["seeds"]) == expected_count
    if cfg["output"]["save_video"]:
        assert len(list(directory.glob("*.mp4"))) == expected_count
    if cfg["perturbation"]["enabled"]:
        patches = np.load(directory / "perturbation.npz")
        assert patches["noise"].shape == (expected_count, 224, 224, 3)
        assert np.isfinite(patches["noise"]).all()
    return manifest, metrics, traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    prefix = "smoke" if args.smoke else f"seed{args.seed}"
    count = 2 if args.smoke else 50
    clean = read(args.root / f"{prefix}_clean", count)
    noisy = read(args.root / f"{prefix}_noisy", count)
    for key in ("dataset_rows", "episodes", "start_steps", "checkpoint_sha256"):
        assert clean[0][key] == noisy[0][key], key
    assert clean[1]["seeds"] == noisy[1]["seeds"]
    result = {"status": "passed", "clean": clean[1], "noisy": noisy[1]}
    if args.smoke:
        zero = read(args.root / "smoke_zero", count)
        assert zero[0]["dataset_rows"] == clean[0]["dataset_rows"]
        assert zero[1]["seeds"] == clean[1]["seeds"]
        assert np.array_equal(zero[2]["successes"], clean[2]["successes"])
        assert np.array_equal(zero[2]["active"], clean[2]["active"])
        assert np.allclose(zero[2]["actions"], clean[2]["actions"], rtol=0, atol=1e-6, equal_nan=True)
        result["zero_noise_matches_clean"] = True
    target = args.root / f"{prefix}_check.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

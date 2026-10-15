"""CPU-only diagnostics of paper equations using existing final checkpoints.

Synthetic inputs probe implementation properties, not navigation performance.
This script never writes a checkpoint or modifies model parameters.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_abs(tensor):
    return float(tensor.detach().abs().max())


def checkpoint_probe(root, name):
    import stable_worldmodel as swm

    path = root / "checkpoints" / name / "weights_epoch_20.pt"
    cfg = OmegaConf.load(path.parent / "config.yaml")
    model = swm.wm.utils.load_pretrained(str(path)).cpu().float().eval()
    model.requires_grad_(False)
    torch.manual_seed(20261015)
    batch, horizon, dim = 2, 5, 192
    hist = torch.randn(batch, 1, 1, dim)
    noise = torch.randn(batch, horizon, 1, dim)
    target = torch.randn_like(noise)
    actions = torch.randn(batch, horizon, int(cfg.model.action_encoder.input_dim))
    changed = actions.clone()
    changed[:, -1] += 0.25
    act_emb, changed_emb = model.action_encoder(actions), model.action_encoder(changed)
    source = model._flow_source_from_noise(hist, noise)
    expected_source = noise * model.flow_source_noise_scale
    if model.flow_source == "noisy_current":
        expected_source = expected_source + hist[:, -1:].expand_as(noise)
    time = torch.full((batch,), 0.375)
    state = 0.625 * source + 0.375 * target

    # Hooks inspect the existing six-block network without truncating it.
    def velocity_with_blocks(emb):
        states = []
        handles = [
            layer.register_forward_hook(
                lambda module, inputs, output: states.append(output.detach().clone())
            )
            for layer in model.predictor.layers
        ]
        try:
            velocity = model.predictor(state, hist, emb, time)
        finally:
            for handle in handles:
                handle.remove()
        return velocity, states

    with torch.no_grad():
        velocity, blocks = velocity_with_blocks(act_emb)
        other_velocity, other_blocks = velocity_with_blocks(changed_emb)
        prediction = model.predict(hist, act_emb, noise=noise, num_steps=8)
        other_prediction = model.predict(hist, changed_emb, noise=noise, num_steps=8)
        reference = expected_source.clone()
        for step in range(8):
            step_time = torch.full((batch,), step / 8)
            value = model.predictor(reference, hist, act_emb, step_time)
            value = model._apply_token_module(model.pred_proj, value)
            reference = reference + value / 8

        torch.manual_seed(718)
        loss = model.flow_loss(hist, act_emb, target)
        torch.manual_seed(718)
        loss_noise = torch.randn_like(target)
        loss_source = model._flow_source_from_noise(hist, loss_noise)
        loss_time = torch.rand(batch)
        mix = loss_time[:, None, None, None]
        loss_state = (1 - mix) * loss_source + mix * target
        loss_velocity = model.predictor(loss_state, hist, act_emb, loss_time)
        loss_velocity = model._apply_token_module(model.pred_proj, loss_velocity)
        squared = (loss_velocity - (target - loss_source)).square()
        norm_loss = squared.flatten(1).sum(1).mean()

    differentiable_actions = actions.detach().clone().requires_grad_(True)
    differentiable_emb = model.action_encoder(differentiable_actions)
    value = model.predictor(state, hist, differentiable_emb, time)
    probe_direction = torch.linspace(-1, 1, dim)
    grad = torch.autograd.grad((value[:, 0, 0] * probe_direction).sum(), differentiable_actions)[0]
    return {
        "checkpoint": f"{name}/weights_epoch_20.pt",
        "checkpoint_sha256": sha256(path),
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "predictor_class": type(model.predictor).__module__ + "." + type(model.predictor).__name__,
        "loaded_core_sha256": {
            "jepa.py": sha256(inspect.getfile(type(model))),
            "module.py": sha256(inspect.getfile(type(model.predictor))),
        },
        "source": model.flow_source,
        "source_std": model.flow_source_noise_scale,
        "causal_self_attention": model.predictor.causal_self_attention,
        "source_reference_max_abs_error": max_abs(source - expected_source),
        "euler_reference_max_abs_error": max_abs(prediction - reference),
        "flow_loss_reference_abs_error": abs(float(loss - squared.mean())),
        "literal_squared_norm_divided_by_mse": float(norm_loss / loss),
        "action_prefix_embedding_max_abs_change": max_abs(act_emb[:, :-1] - changed_emb[:, :-1]),
        "velocity_first_token_max_abs_change": max_abs(velocity[:, 0] - other_velocity[:, 0]),
        "prediction_first_token_max_abs_change": max_abs(prediction[:, 0] - other_prediction[:, 0]),
        "velocity_last_token_max_abs_change": max_abs(velocity[:, -1] - other_velocity[:, -1]),
        "first_token_change_after_each_block": [
            max_abs(before[:, 0] - after[:, 0])
            for before, after in zip(blocks, other_blocks)
        ],
        "first_velocity_projection_gradient_last_action_max_abs": max_abs(grad[:, -1]),
        "training_config": OmegaConf.to_container(cfg, resolve=True),
    }


def noise_probe():
    from perturbations.gaussian import add_noise, make_noise_patches

    class RecordingRng:
        def __init__(self):
            self.rng = np.random.default_rng(718)
            self.epsilon = []

        def uniform(self, *args, **kwargs):
            return self.rng.uniform(*args, **kwargs)

        def normal(self, *args, **kwargs):
            value = self.rng.normal(*args, **kwargs)
            self.epsilon.append(value.copy())
            return value

    results = {}
    for radius in (10, 35):
        rng = RecordingRng()
        cfg = OmegaConf.create({"gaussian_noise": {
            "center": "random_non_agent", "radius": radius, "std": 100,
        }})
        patches, centers = make_noise_patches(rng, cfg, [None, None], (2, 224, 224, 3))
        yy, xx = np.mgrid[:224, :224]
        references = []
        for center, epsilon in zip(centers, rng.epsilon):
            kernel = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * radius ** 2))
            references.append(kernel[..., None] * epsilon)
        reference = np.asarray(references, dtype=np.float32)
        pixels = np.full((2, 1, 224, 224, 3), 128, dtype=np.uint8)
        expected_float = np.clip(pixels.astype(np.float32) + reference[:, None], 0, 255)
        actual = add_noise(pixels, patches, (0, 255))
        results[str(radius)] = {
            "kernel_reference_max_abs_error": float(np.max(np.abs(reference - patches))),
            "clipped_uint8_reference_max_abs_error": float(np.max(np.abs(actual.astype(float) - expected_float.astype(np.uint8)))),
            "float_equation_to_uint8_max_quantization_error": float(np.max(np.abs(actual.astype(float) - expected_float))),
        }
    return results


def sigreg_probe():
    from module import SIGReg

    torch.manual_seed(718)
    values = torch.randn(6, 128, 192)
    reg = SIGReg(knots=17, num_proj=1024)
    torch.manual_seed(912)
    actual = reg(values)
    torch.manual_seed(912)
    directions = torch.randn(192, 1024)
    directions /= directions.norm(dim=0)
    projected = values @ directions
    freqs = torch.linspace(0, 3, 17)
    phase = projected[..., None] * freqs
    phi_real = phase.cos().mean(dim=1)
    phi_imag = phase.sin().mean(dim=1)
    normal_cf = torch.exp(-freqs.square() / 2)
    error = ((phi_real - normal_cf).square() + phi_imag.square()) * normal_cf
    expected = (2 * torch.trapezoid(error, freqs, dim=-1) * 128).mean()
    return {"weighted_symmetric_trapezoid_abs_error": abs(float(actual - expected)),
            "code_value": float(actual), "reference_value": float(expected),
            "quadrature_domain": [-3, 3], "positive_knots_including_zero": 17,
            "num_projections": 1024, "batch": 128, "time_positions": 6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.repo.resolve()))
    os.environ["STABLEWM_HOME"] = str(args.data_root.resolve())
    torch.set_num_threads(1)
    names = {
        "tworoom": "flow_tworoom_seed3073",
        "cube": "flow_cube_seed3073",
        "pusht": "flow_pusht_seed3073_init3072",
        "reacher": "flow_reacher_seed3073_init3073",
    }
    output = {
        "diagnostic_only": True, "device": "cpu", "dtype": "float32",
        "input_kind": "synthetic latent history and normalized-scale actions",
        "probe_seed": 20261015, "last_action_additive_perturbation": 0.25,
        "torch_version": torch.__version__, "script_sha256": sha256(__file__),
        "models": {},
    }
    for task, name in names.items():
        output["models"][task] = checkpoint_probe(args.data_root, name)
        print(f"Completed {task}", flush=True)
    output["rgb_noise_equation17"] = noise_probe()
    output["sigreg"] = sigreg_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

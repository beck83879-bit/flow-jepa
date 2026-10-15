"""Validate the paper profile and the checkpoint produced by a short training run."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf


def check_config(cfg, task, seed, smoke=False):
    pretrained = task in ('pusht', 'reacher')
    frozen = task == 'pusht'
    filename = {'pusht':'pusht_expert_train.h5', 'reacher':'reacher.h5',
                'tworoom':'tworoom.h5', 'cube':'cube_single_expert.h5'}[task]
    expected = {
        'seed':seed, 'trainer.devices':1, 'trainer.max_epochs':1 if smoke else 20,
        'trainer.precision':'bf16-mixed', 'trainer.gradient_clip_val':1.0,
        'loader.batch_size':128, 'data.dataset.num_steps':6, 'data.dataset.frameskip':5,
        'data.dataset.name':str(Path(os.environ['STABLEWM_HOME'])/filename),
        'history_size':1, 'num_preds':5, 'num_flow_steps':8, 'img_size':224, 'embed_dim':192,
        'flow_matching.source':'noisy_current' if pretrained else 'standard_noise',
        'flow_matching.source_noise_scale':0.5 if pretrained else 1.0,
        'model.predictor.causal_self_attention':pretrained,
        'model.predictor.dropout':0.0 if frozen else 0.1,
        'model.predictor.depth':6, 'model.predictor.heads':16, 'model.predictor.dim_head':64,
        'model.predictor.mlp_dim':2048, 'model.encoder.pretrained':False,
        'model.encoder.size':'tiny', 'model.encoder.patch_size':14,
        'pretrained_visual_encoder.enabled':pretrained,
        'loss.sigreg.weight':0.09 if frozen else 0.1,
        'loss.sigreg.kwargs.num_proj':1024, 'loss.sigreg.kwargs.knots':17,
        'optimizer.type':'AdamW', 'optimizer.lr':5e-5, 'optimizer.weight_decay':1e-3,
        'wandb.enabled':False,
    }
    if pretrained:
        expected.update({'pretrained_visual_encoder.freeze':frozen,
                         'pretrained_visual_encoder.load_projector':True,
                         'pretrained_visual_encoder.strict':True})
    for key, value in expected.items():
        actual = OmegaConf.select(cfg, key)
        if actual != value:
            raise ValueError(f'{key}: expected {value!r}, got {actual!r}')
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('pusht','tworoom','reacher','cube'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke-run')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not args.smoke_run:
        with hydra.initialize_config_dir(config_dir=str(Path(__file__).parent/'config/train'),version_base=None):
            cfg = hydra.compose(config_name=f'paper_{args.task}', overrides=[f'seed={args.seed}'])
        checked = check_config(cfg,args.task,args.seed)
        result = {'status':'passed','checked':checked,'config':OmegaConf.to_container(cfg,resolve=True)}
        if cfg.pretrained_visual_encoder.enabled:
            path = Path(cfg.pretrained_visual_encoder.checkpoint)
            result['initialization_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        directory = Path(os.environ['STABLEWM_HOME'])/'checkpoints'/args.smoke_run
        cfg = OmegaConf.load(directory/'config.yaml')
        check_config(cfg,args.task,args.seed,smoke=True)
        assert cfg.model.freeze_encoder == cfg.model.freeze_projector == (args.task=='pusht')
        assert cfg.model.action_encoder.input_dim == (25 if args.task=='cube' else 10)
        path = directory/'weights_epoch_1.pt'
        state = torch.load(path,map_location='cpu',weights_only=True)
        assert all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values())
        model = swm.wm.utils.load_pretrained(str(path)).cpu().eval()
        source_dir = Path(__file__).resolve().parent
        model_file = Path(inspect.getfile(type(model))).resolve()
        predictor_file = Path(inspect.getfile(type(model.predictor))).resolve()
        assert model_file == source_dir / 'jepa.py', model_file
        assert predictor_file == source_dir / 'module.py', predictor_file
        assert type(model.predictor).__name__ == 'FlowTrajectoryPredictor'
        flow_keys = [key for key in state if key.startswith('predictor.time_mlp.')]
        assert len(flow_keys) == 4, flow_keys
        model.set_flow_seed(42)
        assert model.flow_source == cfg.flow_matching.source
        assert model.num_flow_steps == 8
        frozen_tensors = 0
        if args.task == 'pusht':
            before = torch.load(cfg.pretrained_visual_encoder.checkpoint,map_location='cpu',weights_only=True)
            for key, value in state.items():
                if key.startswith(('encoder.', 'projector.')):
                    assert torch.equal(value,before[key]), key
                    frozen_tensors += 1
            model.train()
            assert not model.encoder.training and not model.projector.training
            assert all(not p.requires_grad for p in model.encoder.parameters())
            assert all(not p.requires_grad for p in model.projector.parameters())
        result = {'status':'passed','checkpoint':str(path),'tensor_count':len(state),
                  'strict_reload':True,'finite':True,'unchanged_frozen_tensors':frozen_tensors,
                  'model_file':str(model_file),'predictor_file':str(predictor_file),
                  'predictor_class':type(model.predictor).__name__,'flow_parameter_keys':flow_keys}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()

"""Jointly fine-tune the existing P4 encoder and initialized joint-memory decoder."""
import argparse
import importlib.util
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from tqdm import tqdm

from models.P4JointMemory.P4JointFineTune import P4JointFineTune
from models.P4JointMemory.P4JointMemory import reset_state, detach_state
from run.train_p4_joint_memory import resolve, augment_tokens, losses, pose_errors
from run.utils.load_config import load_config
from run.utils.p4_sequence_data import load_frontend
from run.utils.p4_finetune_data import prepare_input_index, load_streams, make_loader
from run.utils.set_seed import set_seed

ROOT = Path(__file__).resolve().parents[1]


def sub_batches(batch, length, device):
    for start in range(0, batch['valid'].shape[1], length):
        if batch['valid'][:, start:start+length].any():
            yield start, {key:value[:, start:start+length].to(device, non_blocking=True)
                          for key, value in batch.items() if torch.is_tensor(value) and key != 'reset'}


def encode(model, batch, device):
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        return model.encode(batch)


@torch.inference_mode()
def evaluate(model, loader, device, config):
    model.eval()
    state = None
    totals = {name:torch.zeros(2, 3, device=device) for name in ('all', 'cold', 'warm')}
    counts = dict.fromkeys(totals, 0)
    direction = torch.zeros(5, dtype=torch.long, device=device)
    groups = {}
    frames = sum(len(stream['target']) for stream in loader.dataset.streams)
    with tqdm(total=frames, desc='joint validation', unit='frame', dynamic_ncols=True) as progress:
        for batch in loader:
            state = reset_state(state, batch['reset'].to(device))
            for _, sub in sub_batches(batch, config['chunk_length'], device):
                pose, state = model.decoder(encode(model, sub, device), state)
                target, valid = sub['target'], sub['valid']
                error = torch.stack((pose_errors(sub['baseline'], target), pose_errors(pose, target)))
                for name, mask in dict(all=valid, cold=valid & (sub['position'] < config['warmup_frames']),
                                        warm=valid & (sub['position'] >= config['warmup_frames'])).items():
                    totals[name] += error[:, mask].sum(1)
                    counts[name] += int(mask.sum())
                for lane, group_name in enumerate(batch['group']):
                    if valid[lane].any():
                        group = groups.setdefault(group_name, [torch.zeros(2, 3, device=device), 0])
                        group[0] += error[:, lane, valid[lane]].sum(1)
                        group[1] += int(valid[lane].sum())
                torso = target[..., [5, 6], :].mean(-2)-target[..., [11, 12], :].mean(-2)
                truth = target[..., 0, :2]-target[..., [15, 16], :2].mean(-2)
                horizontal = valid & (torso[..., 2].abs()/torso.norm(dim=-1).clamp_min(1e-6) < .4) & (truth.norm(dim=-1) > .5)
                wrong = []
                for current in (sub['baseline'], pose):
                    vector = current[..., 0, :2]-current[..., [15, 16], :2].mean(-2)
                    wrong.append(horizontal & (F.cosine_similarity(vector, truth, dim=-1) < -.5))
                a, b = wrong
                direction += torch.stack((horizontal.sum(), a.sum(), b.sum(), (a & ~b).sum(), (~a & b).sum()))
                progress.set_postfix(mpjpe_cm=f"{100*float(totals['all'][1, 0])/counts['all']:.2f}", refresh=False)
                progress.update(int(valid.sum()))
    def report(total, count):
        return dict(count=count, **{name:dict(zip(('mpjpe', 'relative_mpjpe', 'root_error'), (total[i]/count).tolist()))
                                   for i,name in enumerate(('baseline', 'decoded'))}) if count else dict(count=0)
    return {**{name:report(total, counts[name]) for name,total in totals.items()},
            'groups':{name:report(*v) for name,v in groups.items()},
            'direction':dict(zip(('horizontal_frames', 'baseline_opposite', 'decoded_opposite',
                                  'removed_opposite', 'introduced_opposite'), direction.tolist()))}


def train_epoch(model, loader, optimizer, device, config):
    model.train()
    state = None
    total, count, missing, steps = 0., 0, 0, 0
    norms = dict(decoder=0., transformer=0., point_encoder=0.)
    parameter_groups = dict(decoder=list(model.decoder.parameters()),
                            transformer=list(model.transformer.parameters())+list(model.pos_embedding.parameters()),
                            point_encoder=list(model.tube_embedding.parameters()))
    frames = sum(len(stream['target']) for stream in loader.dataset.streams)
    with tqdm(total=frames, desc='joint training', unit='frame', dynamic_ncols=True) as progress:
        for batch in loader:
            state = reset_state(state, batch['reset'].to(device))
            batch = {key:value.to(device, non_blocking=True) for key,value in batch.items()
                     if torch.is_tensor(value) and key != 'reset'}
            encoded = encode(model, batch, device)
            observed, dropped = augment_tokens(encoded, batch['valid'], config)
            missing += dropped
            pose, state = model.decoder(observed, state)
            state = detach_state(state)
            loss, error, size = losses(pose, batch['target'], batch['valid'], config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint fine-tuning loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameters in parameter_groups.items():
                squared = sum((p.grad.detach().float().square().sum() for p in parameters if p.grad is not None),
                              torch.zeros((), device=device))
                norms[name] += float(squared.sqrt())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(error.detach()); count += size; steps += 1
            progress.set_postfix(mpjpe_cm=f'{100*total/count:.2f}', refresh=False)
            progress.update(size)
    return dict(train_augmented_mpjpe=total/count, train_frames=count, missing_train_frames=missing,
                optimizer_steps=steps, gradient_norms={name:value/steps for name,value in norms.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'run/p4_joint_finetune.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if 'backprop_frames' in config:
        raise ValueError('Remove backprop_frames: decoder gradients now span chunk_length; '
                         'use encoder_batch_size to control encoder memory')
    positive = ('batch_size', 'chunk_length', 'encoder_batch_size', 'epochs', 'input_batch_size',
                'missing_block_max_frames', 'early_stopping_patience')
    if any(type(config[k]) is not int or config[k] < 1 for k in positive):
        raise ValueError('Batch, encoder microbatch, epoch and input sizes must be positive integers')
    if not 0 <= config['encoder_warmup_epochs'] < config['epochs'] or config['num_workers'] < 0:
        raise ValueError('Require nonnegative worker count and encoder warmup shorter than training')
    if not 0 <= config['warmup_frames'] < config['chunk_length']:
        raise ValueError('Invalid validation warmup length')
    if type(config['checkpoint_encoder']) is not bool:
        raise ValueError('checkpoint_encoder must be bool')
    if any(not 0 <= config[k] <= 1 for k in ('point_token_dropout', 'missing_block_probability')):
        raise ValueError('Occlusion probabilities must be in [0,1]')
    if min(config[k] for k in ('learning_rate', 'head_learning_rate', 'encoder_learning_rate')) <= 0:
        raise ValueError('Learning rates must be positive')
    if config['horizontal_pose_weight'] < 1 or config['direction_loss_weight'] < 0:
        raise ValueError('Invalid loss weights')
    config['smoke'] = args.smoke
    if args.smoke:
        config.update(epochs=2, encoder_warmup_epochs=1)
    set_seed(config['seed'])
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    manifest, datasets = prepare_input_index(config)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    source = resolve(config['init_experiment'])
    initial = torch.load(source/config['init_checkpoint'], map_location='cpu', weights_only=False)
    if (initial.get('training_mode') or resolve(initial['config']['baseline_experiment']) != resolve(config['baseline_experiment'])
            or initial['config']['checkpoint'] != config['checkpoint']):
        raise ValueError('Initialization must be the matching frozen-P4 joint-memory experiment')
    frontend, _, encoder_architecture = load_frontend(resolve(config['baseline_experiment']),
        resolve(config['baseline_experiment'])/'checkpoint'/config['checkpoint'], device)
    spec = importlib.util.spec_from_file_location('initialized_joint_decoder', source/'P4JointMemory.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    decoder = module.P4JointMemory(**initial['architecture']).to(device)
    decoder.load_state_dict(initial['model_state_dict'])
    model = P4JointFineTune(frontend, decoder, config['encoder_batch_size'], config['checkpoint_encoder']).to(device)
    del frontend
    head_names = ('head_norm.', 'head_hidden.', 'joint_weight', 'joint_bias')
    optimizer = torch.optim.AdamW([
        dict(params=[p for name,p in decoder.named_parameters() if name.startswith(head_names)], lr=config['head_learning_rate']),
        dict(params=[p for name,p in decoder.named_parameters() if not name.startswith(head_names)], lr=config['learning_rate']),
        dict(params=list(model.transformer.parameters())+list(model.pos_embedding.parameters())+list(model.tube_embedding.parameters()),
             lr=config['encoder_learning_rate'])], weight_decay=config['weight_decay'])
    streams = {mode:load_streams(manifest[mode]) for mode in ('train', 'val')}
    loaders = {mode:make_loader(streams[mode], config, device, mode == 'train', datasets[mode]) for mode in streams}
    output = resolve(config['output_dir'])/(('smoke_' if args.smoke else '')+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output/'model_config.yaml').write_text(yaml.safe_dump(initial['architecture']))
    (output/'frontend_config.yaml').write_text(yaml.safe_dump(encoder_architecture))
    (output/'evidence_manifest.json').write_text(json.dumps(manifest, indent=2))
    torch.save(initial, output/'init_checkpoint.pth')
    for origin in (source/'P4JointMemory.py', resolve(config['baseline_experiment'])/'config/P4Transformer.py',
                   ROOT/'models/P4JointMemory/P4JointFineTune.py', Path(__file__)):
        shutil.copy2(origin, output/origin.name)
    history = [dict(epoch=0, **evaluate(model, loaders['val'], device, config))]
    if not args.smoke and abs(history[0]['all']['decoded']['mpjpe']-initial['best_metric']) > .0005:
        raise ValueError('Initial online metric differs from the source checkpoint by more than 0.5 mm')
    best, stale = history[0]['all']['decoded']['mpjpe'], 0
    def save(name, epoch):
        torch.save(dict(epoch=epoch, training_mode='joint_finetune', model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), architecture=initial['architecture'],
            encoder_architecture=encoder_architecture, config=config, best_metric=best,
            source_epoch=initial['epoch']), output/name)
    save('best.pth', 0)
    (output/'history.json').write_text(json.dumps(history, indent=2))
    print(f"Output: {output}\nInitialized source epoch {initial['epoch']}: {history[0]['all']}", flush=True)
    for epoch in range(1, config['epochs']+1):
        if epoch == config['encoder_warmup_epochs']+1:
            stale = 0
        model.set_encoder_stage(epoch > config['encoder_warmup_epochs'])
        started = time.monotonic()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        print(f"Epoch {epoch}/{config['epochs']}", flush=True)
        training = train_epoch(model, loaders['train'], optimizer, device, config)
        if training['train_frames'] != sum(len(s['target']) for s in streams['train']):
            raise ValueError('Training must supervise each source frame exactly once per epoch')
        report = dict(epoch=epoch, point_encoder_trainable=epoch > config['encoder_warmup_epochs'], **training,
                      **evaluate(model, loaders['val'], device, config), elapsed_seconds=time.monotonic()-started,
                      peak_gpu_memory_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0.)
        history.append(report)
        score = report['all']['decoded']['mpjpe']
        if score < best:
            best, stale = score, 0
            save('best.pth', epoch)
        else:
            stale += 1
        save('last.pth', epoch)
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch}: train={training['train_augmented_mpjpe']:.6f} | {report['all']} | "
              f"direction={report['direction']} | gradients={training['gradient_norms']} | "
              f"steps={training['optimizer_steps']} | peak_gpu={report['peak_gpu_memory_gb']:.2f} GB | "
              f"seconds={report['elapsed_seconds']:.1f}", flush=True)
        # Ensure the second stage is actually tried before stopping for lack of improvement.
        if epoch > config['encoder_warmup_epochs'] and stale >= config['early_stopping_patience']:
            break


if __name__ == '__main__':
    main()

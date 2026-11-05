"""Diagnose whether P4 loses head/feet identity after a correct transition.

GT supplies offline episode labels only; nothing is relabelled or deleted.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from run.analyze_pose_refiner import describe_conditions, head_foot_direction


def audit_stream(stream, group, stream_id):
    target, baseline = [stream[key].float().numpy() for key in ('target', 'baseline')]
    quality = stream['quality'].double().numpy()
    conditions = describe_conditions(target, quality)
    gt_vector = target[:, 0]-target[:, [15, 16]].mean(1)
    pred_vector = baseline[:, 0]-baseline[:, [15, 16]].mean(1)
    horizontal = ((conditions['posture'] == 'horizontal_torso')
                  & (np.linalg.norm(gt_vector[:, :2], axis=-1) > .5))
    cosine = (head_foot_direction(target)*head_foot_direction(baseline)).sum(-1)
    torso = target[:, [5, 6]].mean(1)-target[:, [11, 12]].mean(1)
    vertical = np.abs(torso[:, 2])/np.maximum(np.linalg.norm(torso, axis=1), 1e-6)
    transition = ((gt_vector*pred_vector).sum(-1)
                  / np.maximum(np.linalg.norm(gt_vector, axis=-1)*np.linalg.norm(pred_vector, axis=-1), 1e-6))
    changes = np.flatnonzero(np.diff(np.r_[False, horizontal, False]))
    counts, episodes = Counter(), []
    for begin, end in zip(changes[::2], changes[1::2]):
        if end-begin < 8:
            continue
        current = cosine[begin:end]
        wrong, correct = current < -.5, current > .5
        initial = current[:8]
        if not wrong.any():
            kind = 'no_wrong'
        elif (initial > .5).mean() >= .75:
            kind = 'initial_correct_then_wrong'
        elif (initial < -.5).mean() >= .75:
            kind = 'initial_wrong'
        else:
            kind = 'initial_uncertain'
        previous_correct = np.r_[False, np.maximum.accumulate(correct)[:-1]]
        streak = 0
        established = False
        previous_stable = np.zeros(len(correct), dtype=bool)
        for index, flag in enumerate(correct):
            previous_stable[index] = established
            streak = streak+1 if flag else 0
            established |= streak >= 3
        counts.update(horizontal_frames=len(current), wrong_frames=int(wrong.sum()),
                      wrong_after_prior_correct=int((wrong & previous_correct).sum()),
                      wrong_after_stable_correct=int((wrong & previous_stable).sum()))
        counts[kind] += 1
        history = np.arange(max(0, begin-24), begin)
        history = history[(vertical[history] < .85) & (np.linalg.norm(gt_vector[history, :2], axis=-1) > .25)]
        correct_transition = bool(len(history) and (transition[history] > .5).any())
        if correct_transition:
            counts['wrong_with_correct_transition'] += int(wrong.sum())
        error = np.linalg.norm(baseline[begin:end]-target[begin:end], axis=-1).mean(-1)
        timestamps = stream.get('timestamp')
        episodes.append(dict(group=group, stream=stream_id, begin=int(begin), end=int(end), kind=kind,
                             wrong_frames=int(wrong.sum()), wrong_after_prior_correct=int((wrong & previous_correct).sum()),
                             previous_upright=bool(np.any(vertical[max(0, begin-24):begin] > .7)),
                             correct_transition=correct_transition, mpjpe=float(error.mean()), frames=int(end-begin),
                             timestamp=float(timestamps[begin]) if timestamps is not None else None))
    return counts, episodes


def audit_transitions(manifest):
    report = {}
    for mode, groups in manifest.items():
        counts, episodes = Counter(), []
        for group, info in groups.items():
            for stream_id, stream in enumerate(torch.load(info['cache'], map_location='cpu', weights_only=False)):
                current, rows = audit_stream(stream, group, stream_id)
                counts.update(current)
                episodes.extend(rows)
        report[mode] = dict(summary=dict(counts), episodes=sorted(episodes, key=lambda e:e['wrong_frames'], reverse=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.experiment/'evidence_manifest.json').read_text())
    report = audit_transitions(manifest)
    path = args.experiment/'transition_audit.json'
    path.write_text(json.dumps(report, indent=2))
    print({mode: result['summary'] for mode, result in report.items()})
    print(f'Transition audit: {path}')


if __name__ == '__main__':
    main()

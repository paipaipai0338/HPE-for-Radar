"""Human-readable group metrics: Raw and Guard on separate lines, all / inside."""
from datetime import datetime, timedelta, timezone
import re


def format_group_metrics(group, label, raw=None, guard=None, *, timestamp=None, model=None):
    """Values are in metres; missing results stay N/A. Use timestamp='' with write_log."""
    if timestamp is None:
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    prefix = f'[{timestamp}] ' if timestamp else ''
    prefix += f'Val group {group} | label={label}'
    if model is not None:
        prefix += f' | model={model}'
    lines = []
    for stage, values in (('raw', raw), ('guard', guard)):
        values = values or {}
        error = lambda key: 'N/A' if values.get(key) is None else f'{values[key]:.6f} m'
        count = lambda key: 'N/A' if values.get(key) is None else str(values[key])
        lines.append(
            f'{prefix} | {stage} MPJPE={error("mpjpe")} / {error("mpjpe_inside")} | '
            f'{stage} centered MPJPE={error("centered_mpjpe")} / {error("centered_mpjpe_inside")} | '
            f'{stage} count={count("count")} / {count("count_inside")}')
    return lines


def parse_group_metric(line, stage='raw'):
    """Read the old single-line format or one selected stage of the new format."""
    match = re.search(r'Val group (\S+) \|', line)
    if match is None:
        return None
    recorded_stage = 'guard' if '| guard MPJPE=' in line else 'raw'
    if recorded_stage != stage:
        return None
    prefix = stage + ' ' if f'| {stage} MPJPE=' in line else ''
    error = re.search(r'\| ' + prefix + r'MPJPE=(N/A|[0-9.eE+\-]+)(?: m)?', line)
    count = re.search(r'\| ' + prefix + r'count=(N/A|\d+)', line)
    if error is None or count is None:
        return None
    label = re.search(r'\| label=(.*?) \|', line)
    return {'group': match[1], 'label': label[1] if label else 'unlabelled',
            'mpjpe': None if error[1] == 'N/A' else float(error[1]),
            'count': None if count[1] == 'N/A' else int(count[1])}

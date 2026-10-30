import json
from pathlib import Path
from typing import Dict, Union


def get_meta_info(
    json_path: Union[Path, str] = Path(__file__).with_name('data description.json'),
    *, mode=None, train_groups=None,
) -> Dict:
    """读取 JSON 的 train/val 清单；mode=None 返回全集，供旧 Dataset 使用。

    train_groups 只能选取训练清单的子集。
    返回 Dict[person_id, List[{'date': str, 'valid_group': List[str]}]]。
    """
    if mode not in (None, 'train', 'val'):
        raise ValueError('mode 必须为 train 或 val')
    with open(json_path, encoding='utf-8') as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError('数据描述必须为包含 train/val 清单的对象')

    meta = {'train': {}, 'val': {}}
    group_owners = {}
    for split in meta:
        if not isinstance(data.get(split), list):
            raise ValueError(f'{split} 必须为记录列表')
        for subject in data[split]:
            if not isinstance(subject, dict):
                raise ValueError(f'{split} 中的记录必须为对象')
            date = subject.get('date')
            person_id = subject.get('person_id')
            groups = subject.get('valid_group')
            if not isinstance(date, str) or len(date) != 8 or not date.isdigit():
                raise ValueError('date 必须为 YYYYMMDD 日期字符串')
            if not isinstance(person_id, str) or not person_id:
                raise ValueError('person_id 必须为非空字符串')
            if not isinstance(groups, list) or any(not isinstance(g, str) or not g for g in groups):
                raise ValueError('valid_group 必须为非空组名字符串的列表')
            for group in groups:
                key = (date, group)
                if key in group_owners:
                    previous_split, previous_person = group_owners[key]
                    raise ValueError(
                        f'数据组重复、划分重叠或人员标识冲突: {date}/{group} '
                        f'({previous_split}/{previous_person}, {split}/{person_id})'
                    )
                group_owners[key] = (split, person_id)
            if groups:
                meta[split].setdefault(person_id, []).append({'date': date, 'valid_group': groups})

    if train_groups is not None:
        if mode is None or not isinstance(train_groups, dict) or not train_groups:
            raise ValueError('train_groups 必须为非空日期到组列表的映射，且必须指定 train/val')
        allowed = {(entry['date'], group) for entries in meta['train'].values()
                   for entry in entries for group in entry['valid_group']}
        for date, groups in train_groups.items():
            if not isinstance(date, str) or not isinstance(groups, list) or not groups or any(
                not isinstance(group, str) or not group for group in groups
            ):
                raise ValueError('train_groups 的日期和组名必须为字符串，组列表不能为空')
            invalid = {group for group in groups if (date, group) not in allowed}
            if invalid:
                raise ValueError(f'train_groups 包含验证组或超出训练清单: {date}/{sorted(invalid)}')
        if mode == 'train':
            selected = {}
            for person_id, entries in meta['train'].items():
                for entry in entries:
                    groups = [g for g in entry['valid_group'] if g in train_groups.get(entry['date'], [])]
                    if groups:
                        selected.setdefault(person_id, []).append({**entry, 'valid_group': groups})
            return selected

    if mode is not None:
        return meta[mode]
    combined = {}
    for split in meta.values():
        for person_id, entries in split.items():
            combined.setdefault(person_id, []).extend(entries)
    return combined


if __name__ == '__main__':
    meta_info = get_meta_info()

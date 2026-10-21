import json
from pathlib import Path
from typing import Dict, Union

def get_meta_info(
    json_path: Union[Path, str] = Path(__file__).with_name('data description.json'),
    *, split_method=None, mode='train', val_person_id='4', train_groups=None,
) -> Dict:
    '''
    默认返回 subjects；person 划分读取 person_split，train_groups 可覆盖训练清单。
    -1 多人组始终归入训练；其余组在 subjects_before 前按人员、之后按固定清单划分。

    Dict[str(person_id), List]
        List[Dict]
            Dict[
                'date': str,
                'valid_group': List[str]
            ]
    '''
    meta_info = {}
    group_owners = {}
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for subject in data['subjects']:
        date = subject['date']
        person_id = subject['person_id']
        valid_groups = subject['valid_group']
        for group in valid_groups:
            key = (date, group)
            if key in group_owners:
                raise ValueError(f'数据组重复或人员标识冲突: {date}/{group}')
            group_owners[key] = person_id
        record = {
            'date': date,
            'valid_group': valid_groups
        }
        
        if person_id in meta_info:
            meta_info[person_id].append(record)
        else:
            meta_info[person_id] = [record]
    if mode not in ('train', 'val'):
        raise ValueError('mode 必须为 train 或 val')
    if split_method == 'person':
        config = data['person_split']
        cutoff = config['subjects_before']
        if not isinstance(cutoff, str) or len(cutoff) != 8 or not cutoff.isdigit():
            raise ValueError('person_split.subjects_before 必须为 YYYYMMDD 日期字符串')
        for key in ('excluded_groups', 'train_groups', 'val_groups'):
            groups_by_date = config[key]
            if not isinstance(groups_by_date, dict) or any(
                not isinstance(date, str) or len(date) != 8 or not date.isdigit()
                or not isinstance(groups, list)
                or any(not isinstance(group, str) or not group for group in groups)
                for date, groups in groups_by_date.items()
            ):
                raise ValueError(f'person_split.{key} 必须为日期到组名列表的映射')
        excluded_val_groups = config['excluded_groups']
        val_groups = config['val_groups']
        val_groups = {
            date: [group for group in groups if group not in excluded_val_groups.get(date, set())]
            for date, groups in val_groups.items()
        }
        with Path(__file__).with_name('person_id_mapping.json').open(encoding='utf-8') as file:
            all_person_ids = {person['person_id'] for person in json.load(file)['people']} | {'-1'}
        if not isinstance(val_person_id, str) or val_person_id not in all_person_ids - {'-1'}:
            raise ValueError('val_person_id 必须是 person_id_mapping.json 中的人员编号字符串')
        # -1 是多人标识，优先于日期清单和验证人员规则。
        selected = {}
        configured_train_groups = config['train_groups']
        for groups_by_date in (configured_train_groups, val_groups):
            for date, groups in groups_by_date.items():
                for group in groups:
                    if date >= cutoff and (date, group) not in group_owners:
                        raise ValueError(f'固定清单缺少 subjects 人员标识: {date}/{group}')
        for date, groups in configured_train_groups.items():
            overlap = {g for g in set(groups) & set(val_groups.get(date, []))
                       if group_owners.get((date, g)) != '-1'}
            if date >= cutoff and overlap:
                raise ValueError(f'person_split 训练集与验证集重叠: {date}/{sorted(overlap)}')
        for person_id, entries in meta_info.items():
            for entry in entries:
                date = entry['date']
                groups = []
                for group in entry['valid_group']:
                    if person_id == '-1':
                        include = mode == 'train'
                    elif group in excluded_val_groups.get(date, []):
                        include = False
                    elif date < cutoff:
                        include = (person_id != val_person_id) if mode == 'train' else (person_id == val_person_id)
                    else:
                        fixed_groups = configured_train_groups if mode == 'train' else val_groups
                        include = group in fixed_groups.get(date, [])
                    if include:
                        groups.append(group)
                if groups:
                    selected.setdefault(person_id, []).append({'date': date, 'valid_group': groups})
        if train_groups is not None:
            if not isinstance(train_groups, dict) or not train_groups:
                raise ValueError('train_groups 必须是非空日期到组列表的映射，且 split_method=person')
            allowed = selected if mode == 'train' else get_meta_info(
                json_path, split_method='person', val_person_id=val_person_id,
            )
            allowed_groups = {(entry['date'], group) for entries in allowed.values()
                              for entry in entries for group in entry['valid_group']}
            for date, groups in train_groups.items():
                if not isinstance(date, str) or not isinstance(groups, list) or not groups or not all(isinstance(group, str) for group in groups):
                    raise ValueError('train_groups 的日期和组名必须为字符串，组列表不能为空')
                invalid = {group for group in groups if (date, group) not in allowed_groups}
                if invalid:
                    raise ValueError(f'train_groups 包含验证组或超出训练清单: {date}/{sorted(invalid)}')
            if mode == 'train':
                selected = {
                    person_id: [
                        {**entry, 'valid_group': [g for g in entry['valid_group']
                                                 if g in train_groups.get(entry['date'], [])]}
                        for entry in entries
                    ] for person_id, entries in selected.items()
                }
                selected = {pid: [e for e in entries if e['valid_group']]
                            for pid, entries in selected.items()}
                selected = {pid: entries for pid, entries in selected.items() if entries}
        meta_info = selected
    elif train_groups is not None:
        raise ValueError('train_groups 必须是非空日期到组列表的映射，且 split_method=person')
    return meta_info

if __name__ == '__main__':
    meta_info = get_meta_info()

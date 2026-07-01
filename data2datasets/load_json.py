import json
from pathlib import Path
from typing import Dict, Union

def get_meta_info(json_path: Union[Path, str]) -> Dict:
    '''
    Dict[str(person_id), List]
        List[Dict]
            Dict[
                'date': str,
                'valid_group': List[str]
            ]
    '''
    meta_info = {}
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for subject in data['subjects']:
        date = subject['date']
        person_id = subject['person_id']
        valid_groups = subject['valid_group']
        record = {
            'date': date,
            'valid_group': valid_groups
        }
        
        if person_id in meta_info:
            meta_info[person_id].append(record)
        else:
            meta_info[person_id] = [record]
    return meta_info

if __name__ == '__main__':
    meta_info = get_meta_info(json_path='/mnt/huawei/data description.json')
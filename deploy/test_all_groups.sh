#!/usr/bin/env bash
# ./deploy/test_all_groups.sh
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
targets=(
    20260912/group_028 20260912/group_029 20260912/group_030
    20260912/group_031 20260912/group_032 20260912/group_046
    20260912/group_049 20260912/group_050 20260912/group_051
    20260912/group_096 20260912/group_097 20260912/group_099
    20260912/group_100 20260912/group_101 20260912/group_102
    20260912/group_105 20260912/group_106 20260912/group_108
    20260912/group_109 20260912/group_110
    20260913/group_011 20260913/group_012 20260913/group_013
    20260913/group_014 20260913/group_015 20260913/group_016
)
failed_groups=()

for target in "${targets[@]}"; do
    eval_date="${target%%/*}"
    group_name="${target#*/}"
    group_path="/mnt/huawei/${eval_date}/data_collection/${group_name}"
    if [[ ! -d "$group_path" ]]; then
        echo "Group directory not found: $group_path" >&2
        failed_groups+=("$target")
        continue
    fi
    echo "===== Testing ${target} ====="
    if ! MPLCONFIGDIR=/tmp INFERENCE_DATE="$eval_date" INFERENCE_GROUP="$group_name" \
        python "$script_dir/Inference_selected_group.py" --no-visualize; then
        failed_groups+=("$target")
    fi
done

if ((${#failed_groups[@]})); then
    echo "Failed groups: ${failed_groups[*]}" >&2
    exit 1
fi

echo "All requested groups completed successfully."

import torch
from tqdm import tqdm

from metrics.detection import get_hungarian_match
from run.utils.plot_fig import plt_fig

def train_one_epoch(model, dataloader, optimizer, metric, device, cfg_task):
    model.train()

    for samples in tqdm(dataloader, total=len(dataloader)):
        # 获取模型输入
        input_key = cfg_task['input']
        model_input = {}
        if 'pc' in input_key:
            model_input['input'] = samples[input_key]['padded'].to(device, non_blocking=True)
            model_input['mask'] = samples[input_key]['mask'].to(device, non_blocking=True)
        else:
            model_input['input'] = samples[input_key].to(device, non_blocking=True)

        # 获取监督对象
        target_key = cfg_task['output']
        gt = {
            'padded': samples[target_key]['padded'].to(device, non_blocking=True),
            'mask': samples[target_key]['mask'].to(device, non_blocking=True),
            # 'bbox': samples[target_key]['bbox'].to(device, non_blocking=True)
        }
        optimizer.zero_grad(set_to_none=True)

        pre = model(model_input)

        loss, _ = metric.calculate_batch(pre, gt)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f'训练 loss 出现 NaN 或 Inf: {loss.item()}'
            )

        loss.backward()
        optimizer.step()

    epoch_metric = metric.epoch_end()

    return epoch_metric, metric

def val_one_epoch(model, dataloader, metric, device, cfg_task, if_plot, fig_path):
    model.eval()
    with torch.no_grad():
        plot_pre = None
        plot_gt = None
        plot_model_input = None
        plot_matches = None
        for samples in tqdm(dataloader, total=len(dataloader)):
            input_key = cfg_task['input']
            model_input = {}
            if 'pc' in input_key:
                model_input['input'] = samples[input_key]['padded'].to(device, non_blocking=True)
                model_input['mask'] = samples[input_key]['mask'].to(device, non_blocking=True)
            else:
                model_input['input'] = samples[input_key].to(device, non_blocking=True)

            # 获取监督对象
            target_key = cfg_task['output']
            gt = {
                'padded': samples[target_key]['padded'].to(device, non_blocking=True),
                'mask': samples[target_key]['mask'].to(device, non_blocking=True),
                # 'bbox': samples[target_key]['bbox'].to(device, non_blocking=True)
            }
            batch_action_gt = samples[target_key].get('action')
            if batch_action_gt is not None:
                gt['action'] = batch_action_gt.to(
                    device,
                    non_blocking=True,
                )

            pre = model(model_input)
            loss, _ = metric.calculate_batch(pre, gt)

            if if_plot and plot_pre is None:
                plot_pre = pre
                plot_gt = gt
                plot_model_input = model_input
                if pre.get('bbox') is not None:
                    plot_matches = get_hungarian_match(
                        pre['bbox'],
                        gt['bbox'],
                        gt['mask'],
                        metric.point_cloud_range,
                        bbox_l1_weight=(
                            metric.matching_bbox_l1_weight
                        ),
                        bbox_iou_weight=(
                            metric.matching_bbox_iou_weight
                        ),
                    )

            if not torch.isfinite(loss):
                raise FloatingPointError(f'验证 loss 出现 NaN 或 Inf: {loss.item()}')
        
        epoch_metric = metric.epoch_end()
        if if_plot:
            plt_fig(
                fig_path,
                plot_pre,
                plot_gt,
                plot_model_input,
                matches=plot_matches,
            )
    return epoch_metric, metric

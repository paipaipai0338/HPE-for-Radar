import math
import torch


PATH = "/home/pai/Huawei/temp/result.pkl"
CHUNK = 512


def rank_auc(score, label):
    score = score.double()
    label = label.bool()
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.double)
    # Average ranks for ties.
    vals, inv, counts = torch.unique(score, return_inverse=True, return_counts=True)
    sums = torch.zeros(vals.numel(), dtype=torch.double).scatter_add_(0, inv, ranks)
    ranks = sums[inv] / counts[inv]
    n1 = label.sum().item()
    n0 = label.numel() - n1
    return ((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def quantiles(x):
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=x.dtype)
    return torch.quantile(x, q).tolist()


d = torch.load(PATH, map_location="cpu", mmap=True, weights_only=False)
valid_flat = d["gt_valid"].reshape(-1)
valid_idx = valid_flat.nonzero().squeeze(1)
B, T = d["gt_valid"].shape[:2]

outputs = {k: [] for k in [
    "mpjpe", "root_err", "pa_proxy", "n_pc", "root_range",
    "joint_d_mean", "joint_d_max", "joint_cov10", "joint_cov20", "joint_cov30",
    "bbox_count20", "z_coverage", "person_pc_d_mean",
]}

for start in range(0, valid_idx.numel(), CHUNK):
    idx = valid_idx[start:start + CHUNK]
    b = torch.div(idx, T, rounding_mode="floor")
    t = idx.remainder(T)
    gt = d["pose_gt"][b, t, 0].float()
    pre = d["pose_pre"][b, t, 0].float()
    mask = d["pc_valid"][b, t]
    pc = d["pc"][b, t, :, :3].float()
    # Both pc and gt are already in the high-radar frame.  The saved
    # high_to_low transform is only for optional visualization in low frame.

    finite = torch.isfinite(pc).all(-1)
    mask = mask & finite
    pc_safe = torch.where(mask[..., None], pc, torch.zeros_like(pc))
    n_pc = mask.sum(-1)

    joint_err = torch.linalg.vector_norm(pre - gt, dim=-1)
    mpjpe = joint_err.mean(-1)
    root_err = torch.linalg.vector_norm(pre.mean(1) - gt.mean(1), dim=-1)
    pre_c = pre - pre.mean(1, keepdim=True)
    gt_c = gt - gt.mean(1, keepdim=True)
    pa_proxy = torch.linalg.vector_norm(pre_c - gt_c, dim=-1).mean(-1)
    root_range = torch.linalg.vector_norm(gt.mean(1), dim=-1)

    dist = torch.cdist(gt, pc_safe)
    dist = dist.masked_fill(~mask[:, None, :], float("inf"))
    joint_d = dist.min(-1).values
    joint_d_mean = joint_d.mean(-1)
    joint_d_max = joint_d.max(-1).values
    cov10 = (joint_d <= .10).float().mean(-1)
    cov20 = (joint_d <= .20).float().mean(-1)
    cov30 = (joint_d <= .30).float().mean(-1)

    lo = gt.min(1).values - .20
    hi = gt.max(1).values + .20
    in_box = ((pc >= lo[:, None]) & (pc <= hi[:, None])).all(-1) & mask
    bbox_count20 = in_box.sum(-1)

    pc_to_joint = dist.min(1).values
    person_mask = (pc_to_joint <= .30) & mask
    person_count = person_mask.sum(-1)
    person_pc_d_mean = (
        pc_to_joint.masked_fill(~person_mask, 0).sum(-1)
        / person_count.clamp_min(1)
    )
    zmin = torch.where(person_mask, pc[..., 2], float("inf")).min(-1).values
    zmax = torch.where(person_mask, pc[..., 2], -float("inf")).max(-1).values
    gt_span = (gt[..., 2].max(-1).values - gt[..., 2].min(-1).values).clamp_min(.1)
    zcov = ((zmax - zmin) / gt_span).clamp(0, 2)
    zcov = torch.where(person_count >= 2, zcov, torch.zeros_like(zcov))

    vals = [mpjpe, root_err, pa_proxy, n_pc.float(), root_range,
            joint_d_mean, joint_d_max, cov10, cov20, cov30,
            bbox_count20.float(), zcov, person_pc_d_mean]
    for key, value in zip(outputs, vals):
        outputs[key].append(value.cpu())

for key in outputs:
    outputs[key] = torch.cat(outputs[key])

severe = outputs["mpjpe"] > .2
print(f"valid={severe.numel()} severe={severe.sum().item()} severe_rate={severe.float().mean().item():.6f}")
print("metric | nonsevere mean/median | severe mean/median | AUROC(score predicts severe)")
directions = {
    "n_pc": -1, "root_range": 1, "joint_d_mean": 1, "joint_d_max": 1,
    "joint_cov10": -1, "joint_cov20": -1, "joint_cov30": -1,
    "bbox_count20": -1, "z_coverage": -1, "person_pc_d_mean": 1,
    "root_err": 1, "pa_proxy": 1,
}
for key, direction in directions.items():
    x = outputs[key]
    a, b = x[~severe], x[severe]
    auc = rank_auc(direction * x, severe)
    print(f"{key:16s} | {a.mean():.5f}/{a.median():.5f} | {b.mean():.5f}/{b.median():.5f} | {auc:.5f}")

print("\nsevere rate by joint_cov20 bin")
cov = outputs["joint_cov20"]
for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, .999), (.999, 1.001)]:
    sel = (cov >= lo) & (cov < hi)
    print(lo, hi, sel.sum().item(), severe[sel].float().mean().item() if sel.any() else math.nan)

print("\nsevere rate by bbox_count20 quartile")
x = outputs["bbox_count20"]
edges = torch.quantile(x, torch.tensor([0, .25, .5, .75, 1.]))
for i in range(4):
    sel = (x >= edges[i]) & ((x <= edges[i+1]) if i == 3 else (x < edges[i+1]))
    print(edges[i].item(), edges[i+1].item(), sel.sum().item(), severe[sel].float().mean().item())

print("\nMPJPE quantiles", quantiles(outputs["mpjpe"]))
print("joint_cov20 quantiles", quantiles(outputs["joint_cov20"]))
print("bbox_count20 quantiles", quantiles(outputs["bbox_count20"]))

# Range-stratified comparison to expose confounding by subject distance.
print("\nrange bin | n | severe rate | cov20 nonsevere/severe | bbox nonsevere/severe")
rng = outputs["root_range"]
for lo, hi in [(0, 2), (2, 3), (3, 4), (4, 5), (5, 100)]:
    s = (rng >= lo) & (rng < hi)
    ns, ss = s & ~severe, s & severe
    if s.any():
        print(f"{lo}-{hi} | {s.sum().item()} | {severe[s].float().mean().item():.5f} | "
              f"{outputs['joint_cov20'][ns].mean().item():.4f}/{outputs['joint_cov20'][ss].mean().item():.4f} | "
              f"{outputs['bbox_count20'][ns].mean().item():.2f}/{outputs['bbox_count20'][ss].mean().item():.2f}")

# Severe samples with excellent coverage and nonsevere samples with poor coverage
# demonstrate whether incompleteness can be a necessary/sufficient explanation.
good_cov = (outputs["joint_cov20"] >= .8) & (outputs["bbox_count20"] >= 40)
poor_cov = (outputs["joint_cov20"] < .5) | (outputs["bbox_count20"] < 10)
print("\ncounterexamples")
print("good coverage n/severe rate/severe n:", good_cov.sum().item(),
      severe[good_cov].float().mean().item(), (good_cov & severe).sum().item())
print("poor coverage n/severe rate/nonsevere n:", poor_cov.sum().item(),
      severe[poor_cov].float().mean().item(), (poor_cov & ~severe).sum().item())

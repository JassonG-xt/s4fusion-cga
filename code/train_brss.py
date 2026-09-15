"""Train BRSS-Fusion on a 4 GB-class GPU with optional structural distillation."""

from __future__ import annotations

import argparse
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from S4Fusion import MambaNet
from modules.cga import cga_conflict_loss, cga_commit_loss
from dataset_manifest import DirectoryPairDataset, dataset_weights
from modules.brss_losses import (
    boundary_fusion_loss,    multiscale_reliability_loss,
    reliability_regularization,
    structural_distillation_loss,
)
from train_gates import (
    ZipPairDataset,
    _adjust_crop_size,
    _parse_crop_size,
    _parse_scales,
    _resolve_device,
    _set_seed,
    fusion_loss,
    scale_consistency_loss,
)


def load_checkpoint(model, path: str, device):
    if not path:
        return
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    print(f"loaded {checkpoint}; missing={len(missing)}, unexpected={len(unexpected)}", flush=True)


def _atomic_save(state, path):
    """Crash-safe save: write to a temp file then rename, so an interruption
    mid-write can never truncate an existing good checkpoint (this box gets
    shut down mid-run)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_model(args, teacher: bool = False):
    depths = tuple(int(v) for v in (args.teacher_depths if teacher else args.depths).split(","))
    return MambaNet(
        depths=list(depths),
        depths_decoder=list(depths),
        ss2d_global_scales=(),
        fusion_global_scales=_parse_scales(args.fusion_scales),
        fusion_global_op=getattr(args, "global_op", "scan"),
        gate_type=args.gate_type,
        dynamic_gate_hidden=args.dynamic_gate_hidden,
        boundary_mode=args.boundary_mode,
        boundary_hidden=args.boundary_hidden,
        boundary_global_context=True,
        use_cga=getattr(args, "use_cga", False),
        cga_hidden=getattr(args, "cga_hidden", 16),
    )


def compute_loss(model, ir, vi, args, teacher=None):
    fused, aux = model(ir, vi, return_aux=True)
    base, _, _, _ = fusion_loss(
        fused, ir, vi, args.w_intensity, args.w_grad, args.w_smooth, args.smooth_k
    )
    loss = base
    pieces = {"base": float(base.detach())}

    boundary_aux = aux.get("boundary", [])
    if boundary_aux:
        boundary = boundary_fusion_loss(fused, ir, vi, boundary_aux)
        regularity = reliability_regularization(boundary_aux)
    else:
        boundary = fused.new_zeros(())
        regularity = fused.new_zeros(())
    loss = loss + args.w_boundary * boundary + args.w_reliability * regularity
    pieces.update(boundary=float(boundary.detach()), reliability=float(regularity.detach()))

    if args.w_scale > 0 or args.w_reliability_scale > 0:
        h, w = _adjust_crop_size((max(8, ir.shape[-2] // 2), max(8, ir.shape[-1] // 2)))
        ir_lr = F.interpolate(ir, (h, w), mode="bilinear", align_corners=False)
        vi_lr = F.interpolate(vi, (h, w), mode="bilinear", align_corners=False)
        fused_lr, aux_lr = model(ir_lr, vi_lr, return_aux=True)
        scale = scale_consistency_loss(fused, fused_lr, edge_weight=0.5)
        if boundary_aux and aux_lr.get("boundary", []):
            rel_scale = multiscale_reliability_loss(boundary_aux, aux_lr["boundary"])
        else:
            rel_scale = fused.new_zeros(())
        loss = loss + args.w_scale * scale + args.w_reliability_scale * rel_scale
        pieces.update(scale=float(scale.detach()), reliability_scale=float(rel_scale.detach()))

    if teacher is not None:
        with torch.no_grad():
            teacher_fused, teacher_aux = teacher(ir, vi, return_aux=True)
        d_out, d_feature, d_boundary = structural_distillation_loss(
            fused, teacher_fused, aux, teacher_aux
        )
        loss = loss + args.w_distill_output * d_out
        loss = loss + args.w_distill_feature * d_feature
        loss = loss + args.w_distill_boundary * d_boundary
        pieces.update(distill_output=float(d_out), distill_feature=float(d_feature), distill_boundary=float(d_boundary))

    cga_aux = aux.get("cga")
    if cga_aux and getattr(args, "w_cga_conflict", 0.0) > 0:
        cga_l = cga_conflict_loss(cga_aux)
        loss = loss + args.w_cga_conflict * cga_l
        pieces["cga_conflict"] = float(cga_l.detach())
    if cga_aux and getattr(args, "w_cga_commit", 0.0) > 0:
        cga_c = cga_commit_loss(cga_aux)
        loss = loss + args.w_cga_commit * cga_c
        pieces["cga_commit"] = float(cga_c.detach())
    return loss, pieces


def main():
    parser = argparse.ArgumentParser(description="Train Boundary-Reliability-aware S4Fusion")
    parser.add_argument("--data-root", default="../dataset")
    parser.add_argument("--train-manifest", default="../dataset/manifests/train_all.csv")
    parser.add_argument("--val-manifest", default="../dataset/manifests/val_all.csv")
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--samples-per-epoch", type=int, default=0, help="sampled crops per epoch; 0 uses the full manifest length")
    parser.add_argument("--train-split", default="", help="legacy ZIP split")
    parser.add_argument("--val-split", default="", help="legacy ZIP split")
    parser.add_argument("--m3fd-zip", default="")
    parser.add_argument("--roadscene-zip", default="")
    parser.add_argument("--default-dataset", default=None)
    parser.add_argument("--checkpoint", required=True, help="pretrained S4Fusion/MS-Global checkpoint")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--save-path", default="checkpoints/brss_fusion.pt")
    parser.add_argument("--last-save-path", default="", help="latest resumable checkpoint; defaults to <save-path>_last")
    parser.add_argument("--resume", default="", help="resume model/optimizer/scaler and epoch from a training checkpoint")
    parser.add_argument("--crop-size", default="128")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train-steps", type=int, default=0, help="debug limit per epoch; 0 uses all batches")
    parser.add_argument("--max-val-steps", type=int, default=0, help="debug limit per epoch; 0 uses all batches")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gate-lr", type=float, default=0.0, help="separate learning rate for CGA gate parameters (0 = use global lr)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depths", default="1,2,1")
    parser.add_argument("--teacher-depths", default="1,2,1")
    parser.add_argument("--fusion-scales", default="4,8")
    parser.add_argument("--global-op", choices=["scan", "pool1x1", "dw3x3"], default="scan",
                        help="fusion global-branch operator; pool1x1/dw3x3 are the H2 external-op controls")
    parser.add_argument("--gate-type", choices=["scalar", "dynamic"], default="dynamic")
    parser.add_argument("--dynamic-gate-hidden", type=int, default=16)
    parser.add_argument("--boundary-mode", choices=["none", "residual", "state", "full"], default="full")
    parser.add_argument("--boundary-hidden", type=int, default=16)
    parser.add_argument("--use-cga", action=argparse.BooleanOptionalAction, default=False,
                        help="enable Conflict-Gated Arbitration head (image-space, zero-init)")
    parser.add_argument("--cga-hidden", type=int, default=16)
    parser.add_argument("--w-cga-conflict", type=float, default=0.1,
                        help="weight for grounding the conflict head on the self-derived conflict signal")
    parser.add_argument("--w-cga-commit", type=float, default=0.0,
                        help="weight for the conflict-gated commit loss: pushes fused output toward the "
                             "salient-modality commitment in conflict regions, which drives the zero-init "
                             "gate open (H5). 0 = disabled (backward compatible).")
    parser.add_argument("--w-intensity", type=float, default=1.0)
    parser.add_argument("--w-grad", type=float, default=10.0)
    parser.add_argument("--w-smooth", type=float, default=0.1)
    parser.add_argument("--smooth-k", type=float, default=10.0)
    parser.add_argument("--w-boundary", type=float, default=2.0)
    parser.add_argument("--w-reliability", type=float, default=0.1)
    parser.add_argument("--w-scale", type=float, default=0.5)
    parser.add_argument("--w-reliability-scale", type=float, default=0.2)
    parser.add_argument("--w-distill-output", type=float, default=1.0)
    parser.add_argument("--w-distill-feature", type=float, default=0.2)
    parser.add_argument("--w-distill-boundary", type=float, default=0.5)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    crop = _adjust_crop_size(_parse_crop_size(args.crop_size))
    if Path(args.train_manifest).is_file() and Path(args.val_manifest).is_file():
        train_set = DirectoryPairDataset(args.data_root, args.train_manifest, crop, True, True)
        val_set = DirectoryPairDataset(args.data_root, args.val_manifest, crop, True, False)
        sampler = None
        samples_per_epoch = args.samples_per_epoch if args.samples_per_epoch > 0 else len(train_set)
        if args.balanced_sampling:
            weights = dataset_weights(args.train_manifest)
            sampler = WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True)
        elif args.samples_per_epoch > 0:
            sampler = RandomSampler(train_set, replacement=True, num_samples=samples_per_epoch)
        train_loader = DataLoader(
            train_set, args.batch_size, shuffle=sampler is None, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        if not all([args.train_split, args.val_split, args.m3fd_zip, args.roadscene_zip]):
            raise FileNotFoundError("manifests not found and legacy ZIP arguments are incomplete")
        train_set = ZipPairDataset(args.train_split, args.m3fd_zip, args.roadscene_zip, crop, args.default_dataset, True, True)
        val_set = ZipPairDataset(args.val_split, args.m3fd_zip, args.roadscene_zip, crop, args.default_dataset, True, False)
        legacy_sampler = None
        if args.samples_per_epoch > 0:
            legacy_sampler = RandomSampler(train_set, replacement=True, num_samples=args.samples_per_epoch)
        train_loader = DataLoader(
            train_set, args.batch_size, shuffle=legacy_sampler is None, sampler=legacy_sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args).to(device)
    load_checkpoint(model, args.checkpoint, device)
    teacher = None
    if args.teacher_checkpoint:
        teacher = build_model(args, teacher=True).to(device).eval()
        load_checkpoint(teacher, args.teacher_checkpoint, device)
        for parameter in teacher.parameters():
            parameter.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.gate_lr > 0:
        gate_params = [
            p for n, p in model.named_parameters()
            if ".cga." in n or "gate_scale" in n or "gate_" in n
        ]
        other_params = [
            p for n, p in model.named_parameters()
            if not (".cga." in n or "gate_scale" in n or "gate_" in n)
        ]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0},
        ])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    autocast = torch.cuda.amp.autocast if scaler.is_enabled() else nullcontext
    best = float("inf")
    start_epoch = 1
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.last_save_path:
        last_save_path = Path(args.last_save_path)
    else:
        suffix = save_path.suffix or ".pt"
        last_save_path = save_path.with_name(f"{save_path.stem}_last{suffix}")
    last_save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_state = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])
        rng_state = resume_state.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"].cpu())
        if device.type == "cuda" and "cuda" in rng_state:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])
        best = float(resume_state.get("best_val_loss", resume_state.get("val_loss", best)))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        print(f"resumed {resume_path} at epoch={start_epoch} best={best:.6f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_batches = 0
        for step, (vi, ir) in enumerate(train_loader, 1):
            vi, ir = vi.to(device, non_blocking=True), ir.to(device, non_blocking=True)
            with autocast():
                loss, _ = compute_loss(model, ir, vi, args, teacher)
                scaled_loss = loss / args.accumulation_steps
            scaler.scale(scaled_loss).backward()
            limit_reached = args.max_train_steps > 0 and step >= args.max_train_steps
            if step % args.accumulation_steps == 0 or step == len(train_loader) or limit_reached:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_total += float(loss.detach())
            train_batches += 1
            if limit_reached:
                break

        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for step, (vi, ir) in enumerate(val_loader, 1):
                vi, ir = vi.to(device, non_blocking=True), ir.to(device, non_blocking=True)
                with autocast():
                    loss, _ = compute_loss(model, ir, vi, args, teacher)
                val_total += float(loss)
                val_batches += 1
                if args.max_val_steps > 0 and step >= args.max_val_steps:
                    break
        train_mean = train_total / max(1, train_batches)
        val_mean = val_total / max(1, val_batches)
        print(f"epoch={epoch} train={train_mean:.6f} val={val_mean:.6f}", flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "val_loss": val_mean,
            "best_val_loss": min(best, val_mean),
            "args": vars(args),
            "rng_state": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        }
        _atomic_save(state, last_save_path)
        if val_mean < best:
            best = val_mean
            state["best_val_loss"] = best
            _atomic_save(state, save_path)
    print(f"best checkpoint: {save_path} (val={best:.6f}); latest: {last_save_path}", flush=True)


if __name__ == "__main__":
    main()

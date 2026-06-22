from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps


PROJ = Path("/cluster/users/grad/2025/25t8103/project")


def load_manifest(path: Path, limit: int | None) -> list[dict]:
    items = json.load(path.open(encoding="utf-8"))
    if limit is not None:
        items = items[:limit]
    return items


def resize_square(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def edit_instruction(item: dict) -> str:
    host = str(item.get("host_tokens") or "").strip()
    new = str(item.get("new_tokens") or "").strip()
    family = item.get("family_label", "")
    if family == "T1_attached_accessory":
        return f"add {new} to the {host}".strip()
    if family == "T2_container_insertion":
        return f"add {new} inside the {host}".strip()
    if family == "T3_surface_decal":
        return f"add {new} on the {host}".strip()
    if family == "T4_local_recolor":
        return f"make the {host} {new}".strip()
    if family == "T5_same_color_material":
        return f"make the {host} a {new}".strip()
    return f"edit the image to match: {item['target_prompt']}"


def ledits_prompt(item: dict) -> str:
    host = str(item.get("host_tokens") or "").strip()
    new = str(item.get("new_tokens") or "").strip()
    family = item.get("family_label", "")
    if family in {"T4_local_recolor", "T5_same_color_material"} and host and new:
        return f"{new} {host}"
    if new:
        return new
    return item["target_prompt"]


def write_artifacts(
    run_dir: Path,
    item: dict,
    method: str,
    result: Image.Image,
    seed: int,
    runtime: float,
    peak_gb: float | None,
    command: str,
    extra: dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    result.save(run_dir / "result.png")
    metadata = {
        "method": method,
        "baseline": method,
        "task": item["key"],
        "seed": seed,
        "image": item["image"],
        "source_image": item["image"],
        "source_prompt": item["source_prompt"],
        "target_prompt": item["target_prompt"],
        "family": item.get("family"),
        "family_label": item.get("family_label"),
        "flowedit_idx": item.get("flowedit_idx"),
        "runtime_seconds": runtime,
        "peak_gpu_memory_gb": peak_gb,
        **extra,
    }
    stats = {"steps": [{"step": f"{method}_complete", "mask_area_ratio": 0.0}]}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")


def run_ip2p(args: argparse.Namespace, items: list[dict]) -> None:
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.ip2p_model,
        torch_dtype=torch.float16 if args.dtype == "float16" else torch.float32,
        safety_checker=None,
        local_files_only=not args.allow_download,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    command = " ".join(shlex.quote(x) for x in ["python", __file__, *args.raw_argv])
    for item in items:
        for seed in args.seeds:
            run_dir = args.output_dir / item["key"] / "instruct_pix2pix" / f"seed_{seed}"
            if args.skip_complete and (run_dir / "result.png").is_file():
                print("skip", run_dir, flush=True)
                continue
            image = resize_square(Path(item["image"]), args.image_size)
            instruction = edit_instruction(item)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            start = time.time()
            result = pipe(
                instruction,
                image=image,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                generator=generator,
            ).images[0]
            runtime = time.time() - start
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
            write_artifacts(
                run_dir,
                item,
                "instruct_pix2pix",
                result,
                seed,
                runtime,
                peak_gb,
                command,
                {"instruction": instruction, "model": args.ip2p_model},
            )
            print("complete instruct_pix2pix", item["key"], seed, f"{runtime:.1f}s", flush=True)


def run_ledits(args: argparse.Namespace, items: list[dict]) -> None:
    try:
        from diffusers import LEditsPPPipelineStableDiffusion as LEditsPipeline
    except ImportError:
        from leditspp import StableDiffusionPipeline_LEDITS as LEditsPipeline

    pipe = LEditsPipeline.from_pretrained(
        args.ledits_model,
        torch_dtype=torch.float16 if args.dtype == "float16" else torch.float32,
        safety_checker=None,
        local_files_only=not args.allow_download,
    )
    pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    command = " ".join(shlex.quote(x) for x in ["python", __file__, *args.raw_argv])
    for item in items:
        for seed in args.seeds:
            run_dir = args.output_dir / item["key"] / "ledits_pp" / f"seed_{seed}"
            if args.skip_complete and (run_dir / "result.png").is_file():
                print("skip", run_dir, flush=True)
                continue
            generator = torch.Generator(device=args.device).manual_seed(seed)
            edit = ledits_prompt(item)
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            start = time.time()
            _ = pipe.invert(
                image_path=item["image"],
                source_prompt=item["source_prompt"],
                source_guidance_scale=args.source_guidance_scale,
                num_inversion_steps=args.inversion_steps,
                skip=args.skip,
                generator=generator,
                verbose=False,
            )
            result = pipe(
                editing_prompt=[edit],
                reverse_editing_direction=[False],
                edit_guidance_scale=[args.edit_guidance_scale],
                edit_threshold=[args.edit_threshold],
                edit_warmup_steps=args.edit_warmup_steps,
                use_cross_attn_mask=args.use_cross_attn_mask,
                use_intersect_mask=args.use_intersect_mask,
                output_type="pil",
            ).images[0]
            runtime = time.time() - start
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
            write_artifacts(
                run_dir,
                item,
                "ledits_pp",
                result,
                seed,
                runtime,
                peak_gb,
                command,
                {"editing_prompt": edit, "model": args.ledits_model},
            )
            print("complete ledits_pp", item["key"], seed, f"{runtime:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJ / "data/flowedit_compatible_118/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJ / "outputs/flowedit_baselines")
    parser.add_argument("--method", choices=("instruct_pix2pix", "ledits_pp"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--ip2p-model", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--ledits-model", default="sd-legacy/stable-diffusion-v1-5")
    parser.add_argument("--inversion-steps", type=int, default=30)
    parser.add_argument("--skip", type=float, default=0.15)
    parser.add_argument("--source-guidance-scale", type=float, default=3.5)
    parser.add_argument("--edit-guidance-scale", type=float, default=8.0)
    parser.add_argument("--edit-threshold", type=float, default=0.75)
    parser.add_argument("--edit-warmup-steps", type=int, default=0)
    parser.add_argument("--use-cross-attn-mask", action="store_true")
    parser.add_argument("--use-intersect-mask", action="store_true", default=True)
    args = parser.parse_args()
    args.raw_argv = __import__("sys").argv[1:]
    args.output_dir = args.output_dir / args.method

    items = load_manifest(args.manifest, args.limit)
    if args.method == "instruct_pix2pix":
        run_ip2p(args, items)
    else:
        run_ledits(args, items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

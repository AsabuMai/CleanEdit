import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = Path(os.environ.get("MANIFEST", str(PROJ / "data/flowedit_compatible_123/manifest.json")))
MODE = os.environ.get("MODE", "flux")
METHOD = f"sam_flow_{MODE}"
OUT = PROJ / os.environ.get("OUT", f"outputs/flowedit_baselines/{METHOD}")
LIMIT = int(os.environ.get("LIMIT", "0"))
SEED = os.environ.get("SEED", "10")
MAX_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "512"))
SAM = PROJ / "_baselines/src/Sam-Flow"
CONFIG = SAM / ("configs/flux.yaml" if MODE == "flux" else "configs/sd3.yaml")


def split_tokens(value: str) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def prepare_image(src: Path, dst: Path) -> Path:
    image = Image.open(src).convert("RGB")
    if max(image.size) > MAX_SIZE:
        scale = MAX_SIZE / float(max(image.size))
        image = image.resize(
            (max(16, int(round(image.width * scale))), max(16, int(round(image.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst)
    return dst


def main() -> None:
    items = json.load(MANIFEST.open())
    if LIMIT:
        items = items[:LIMIT]
    done = 0
    for item in items:
        final = OUT / item["key"] / METHOD / f"seed_{SEED}" / "result.png"
        if final.exists():
            print("skip", item["key"], flush=True)
            continue
        run_dir = final.parent
        tmp = run_dir / "tmp_samflow"
        prepared = prepare_image(Path(item["image"]), tmp / "input_512.png")
        output_root = tmp / f"results_{MODE}"
        target_code = f"{item['key']}_{METHOD}_seed_{SEED}"
        expected = output_root / prepared.stem / target_code / f"{target_code}_output.png"
        cmd = [
            sys.executable,
            str(SAM / "scripts/run_image.py"),
            "--mode",
            MODE,
            "--config",
            str(CONFIG),
            "--image",
            str(prepared),
            "--source-prompt",
            item["source_prompt"],
            "--target-prompt",
            item["target_prompt"],
            "--target-code",
            target_code,
            "--output-root",
            str(output_root),
            "--overwrite",
        ]
        for token in split_tokens(item.get("host_tokens", "")):
            cmd += ["--source-token", token]
        for token in split_tokens(item.get("new_tokens", "")):
            cmd += ["--target-token", token]
        print("RUN", METHOD, item["key"], flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "command.txt").write_text(" ".join(subprocess.list2cmdline([x]) for x in cmd) + "\n")
        proc = subprocess.run(cmd, cwd=str(SAM))
        if proc.returncode == 0 and expected.exists():
            shutil.copy2(expected, final)
            done += 1
            print("OK", item["key"], flush=True)
        else:
            print("FAILED", item["key"], "rc", proc.returncode, "expected", expected, flush=True)
    print("ALLDONE", METHOD, done, "/", len(items), flush=True)


if __name__ == "__main__":
    main()

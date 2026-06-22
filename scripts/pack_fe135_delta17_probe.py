import json
import subprocess
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
MANIFEST = PROJ / "data/flowedit_compatible_135/manifest_sam_17new.json"
LIST = Path("/tmp/fe135_delta17_files.txt")
TAR = Path("/tmp/fe135_delta17_probe_25t8103.tgz")


def rel(path: str) -> str:
    p = Path(path)
    if p.is_absolute() and str(p).startswith(str(PROJ)):
        return str(p.relative_to(PROJ))
    return str(p)


def main() -> None:
    items = json.load(MANIFEST.open(encoding="utf-8"))
    paths = ["data/flowedit_compatible_135/manifest_sam_17new.json"]
    for item in items:
        key = item["key"]
        paths.append(rel(item["image"]))
        paths.append(item["pp_local_mask"])
        paths.append(f"outputs/fe135_delta17_dece_flux/{key}/dece_rf_flux/seed_10/result.png")
        paths.append(f"outputs/fe135_delta17_dece_sd3/{key}/support_v3_controller_rmsgap/seed_10/result.png")
    LIST.write_text("\n".join(paths) + "\n", encoding="utf-8")
    subprocess.run(["tar", "-czhf", str(TAR), "-T", str(LIST)], cwd=PROJ, check=True)
    print(TAR, len(paths))


if __name__ == "__main__":
    main()

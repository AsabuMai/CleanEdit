import json
from pathlib import Path

proj = Path("/cluster/users/grad/2025/25t8103/project")
src = proj / "data/flowedit_compatible_135/manifest_sam_135.json"
dst = proj / "data/flowedit_compatible_135/manifest_t4_recolor_19.json"

manifest = json.loads(src.read_text(encoding="utf-8"))

def is_t4(entry):
    family = str(entry.get("family", ""))
    label = str(entry.get("family_label", ""))
    config = str(entry.get("config", ""))
    return family.startswith("F4") or family.startswith("T4") or label.startswith("T4") or config.startswith("6_")

t4 = [entry for entry in manifest if is_t4(entry)]
if len(t4) != 19:
    raise SystemExit(f"expected 19 T4 recolor tasks, got {len(t4)}")

dst.write_text(json.dumps(t4, indent=2) + "\n", encoding="utf-8")
print(dst)
print(len(t4))

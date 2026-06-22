import json
from pathlib import Path


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
BASE = PROJ / "data/flowedit_compatible_135/manifest.json"
SAM118 = PROJ / "data/flowedit_compatible_118/manifest_sam_all.json"
SAM17 = PROJ / "data/flowedit_compatible_135/manifest_sam_17new.json"
OUT = PROJ / "data/flowedit_compatible_135/manifest_sam_135.json"

MASK_FIELDS = [
    "pp_local_mask",
    "sam_anchor_mask",
    "sam_support_meta",
    "sam_phrase",
    "sam_support_relation",
    "global_mask",
]


def by_key(path: Path) -> dict[str, dict]:
    return {item["key"]: item for item in json.load(path.open(encoding="utf-8"))}


def main() -> None:
    base = json.load(BASE.open(encoding="utf-8"))
    sam = by_key(SAM118)
    sam.update(by_key(SAM17))
    merged = []
    missing = []
    for item in base:
        key = item["key"]
        row = dict(item)
        src = sam.get(key)
        if src is None:
            missing.append(key)
        else:
            for field in MASK_FIELDS:
                if field in src:
                    row[field] = src[field]
        merged.append(row)
    OUT.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)
    print("n", len(merged))
    print("missing_masks", len(missing), missing[:20])
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

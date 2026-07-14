from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = [
    "baseline",
    "task",
    "seed",
    "status",
    "source_image",
    "source_prompt",
    "target_prompt",
    "source_tokens",
    "target_tokens",
    "unchanged_tokens",
    "result_image",
    "metadata",
    "command",
    "matched_conditions",
    "failure_reason",
    "notes",
]


def target_from_key(key: str, current: str) -> str:
    if key.endswith("volkswagen_logo"):
        return "Volkswagen logo"
    if current:
        return current
    return key.rsplit("_", 1)[-1]


def category(key: str) -> str:
    if "_gas_station_" in key:
        return "gas"
    if "_groceries_" in key:
        return "grocery"
    if "_luna_" in key:
        return "luna"
    if "_sign_" in key:
        return "billboard"
    if "_stop_arrow_" in key:
        return "stop_arrow"
    if "_stop_sticker_" in key:
        return "sticker"
    if "_stop_" in key:
        return "stop"
    if "_this_must_be_the_place_" in key:
        return "neon"
    if "_bus_" in key:
        return "bus"
    return "generic"


def token_pair(key: str, target: str) -> tuple[str, str]:
    cat = category(key)
    if cat == "grocery":
        if key.endswith("_1_bacon"):
            return "paper", "bacon,BACON"
        if key.endswith("_2_coffee"):
            return "eggs", "coffee,COFFEE"
        return "bread", f"{target.lower()},{target.upper()}"
    source_by_cat = {
        "gas": "cafe",
        "luna": "luna",
        "billboard": "love",
        "stop_arrow": "stop",
        "sticker": "stop",
        "stop": "stop",
        "neon": "this",
        "bus": "tire",
    }
    source = source_by_cat.get(cat, "")
    norm = target.lower()
    if norm == "volkswagen logo":
        return source, "volkswagen logo,logo"
    if norm == "heart":
        return source, "heart"
    return source, f"{norm},{target.upper()}"


def prompt_suffix(key: str, target: str) -> str:
    cat = category(key)
    exact = target if target == "Volkswagen logo" else target.upper()
    common = (
        f'The edited target word must read exactly "{exact}" with no extra letters, '
        "no old-letter residue, and no pasted overlay."
    )
    if cat == "gas":
        return common + " Keep the same gas station sign board, cars, canopy, and background."
    if cat == "grocery":
        return common + " Keep the same brown paper, magnet, handwriting style, and list layout."
    if cat == "luna":
        return common + " Keep the same neon sign frame, glow, street view, and perspective."
    if cat == "billboard":
        return common + " Keep the same billboard plane, lamps, buildings, and perspective."
    if cat in {"stop", "stop_arrow"}:
        return common + " Keep the same octagonal red sign shape, pole, background, and arrow sign if present."
    if cat == "sticker":
        return common + " Keep the same sticker cutline, comic burst, white border, and shadow."
    if cat == "neon":
        return common + " Keep the same glass neon tubes, glow, wall, brackets, and perspective."
    if cat == "bus":
        return (
            "The van front must show a clear Volkswagen logo between the headlights. "
            "Keep the same van, house, driveway, trees, and camera view."
        )
    return common


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build failed-only FE135 T3 Sam-Flow probe v2 manifest.")
    ap.add_argument("--v1-manifest", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v1_manifest.csv"))
    ap.add_argument("--v1-audit", type=Path, default=Path("fe135_t3_samflow_probe_v1_review_all/strict_visual_audit_probe.csv"))
    ap.add_argument("--out", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v2_manifest.csv"))
    args = ap.parse_args()

    manifest_rows = read_csv(args.v1_manifest)
    by_pair = {
        (("sam_flow_sd3_probe_v1" if row["baseline"] == "sam_flow_sd3" else "sam_flow_flux_probe_v1"), row["task"]): row
        for row in manifest_rows
    }
    audit_rows = read_csv(args.v1_audit)

    out_rows: list[dict[str, str]] = []
    for audit in audit_rows:
        if audit["status"] != "fail":
            continue
        source = by_pair[(audit["method"], audit["key"])]
        target = target_from_key(source["task"], source.get("target_tokens", ""))
        source_tokens, target_tokens = token_pair(source["task"], target)
        row = {name: source.get(name, "") for name in FIELDNAMES}
        row.update(
            {
                "status": "pending",
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "unchanged_tokens": "",
                "target_prompt": f"{source['target_prompt']} {prompt_suffix(source['task'], target)}",
                "result_image": "",
                "metadata": "",
                "command": "",
                "matched_conditions": "",
                "failure_reason": "",
                "notes": (
                    "FE135 T3 Sam-Flow probe_v2 failed-only prompt/token retry; "
                    "fixed seed10; separate probe, not final candidate; "
                    f"v1_failure_type={audit['failure_type']}; v1_note={audit['note']}"
                ),
            }
        )
        out_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.out)
    print("rows", len(out_rows))
    counts: dict[str, int] = {}
    for row in out_rows:
        counts[row["baseline"]] = counts.get(row["baseline"], 0) + 1
    for key in sorted(counts):
        print(key, counts[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

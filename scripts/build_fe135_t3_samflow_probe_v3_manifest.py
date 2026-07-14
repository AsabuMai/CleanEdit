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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def letter_hint(target: str) -> str:
    if target == "heart":
        return "a single clean heart symbol"
    if target == "Volkswagen logo":
        return "a clean Volkswagen logo"
    return " ".join(target.upper())


def target_from_key(key: str) -> str:
    return {
        "fe_120_gas_station_3_eccv": "ECCV",
        "fe_121_gas_station_4_food": "FOOD",
        "fe_131_groceries_1_bacon": "BACON",
        "fe_133_groceries_3_cvpr": "CVPR",
        "fe_134_groceries_4_eccv": "ECCV",
        "fe_135_groceries_5_iccv": "ICCV",
        "fe_167_luna_4_eccv": "ECCV",
        "fe_170_luna_7_heart": "heart",
        "fe_223_sign_2_eccv": "ECCV",
        "fe_224_sign_3_iccv": "ICCV",
        "fe_245_stop_sticker_2_eccv": "ECCV",
        "fe_246_stop_sticker_3_iccv": "ICCV",
        "fe_166_luna_3_iccv": "ICCV",
        "fe_236_stop_2_iccv": "ICCV",
        "fe_240_stop_arrow_3_eccv": "ECCV",
    }[key]


def tune(key: str) -> tuple[str, str, str]:
    target = target_from_key(key)
    target_tokens = f"{target.lower()},{target.upper()},text,letters" if target != "heart" else "heart,symbol"
    spelled = letter_hint(target)
    if "_gas_station_" in key:
        return (
            "cafe,CAFE",
            target_tokens,
            (
                f'The gas station sign must read exactly "{target}" as large red capital letters, spelled {spelled}. '
                "Remove CAFE completely, do not leave old strokes, and keep the same sign board, canopy, cars, and desert background."
            ),
        )
    if "_groceries_" in key:
        if key.endswith("_1_bacon"):
            source = "bread,BREAD"
            item = "first list item"
        elif key.endswith("_2_coffee"):
            source = "eggs,EGGS"
            item = "second list item"
        else:
            source = "bread,BREAD"
            item = "first list item"
        return (
            source,
            target_tokens,
            (
                f'Replace only the {item} with "{target}", spelled {spelled}, in the same marker handwriting. '
                "Keep the brown paper, magnet, hand, EGGS and MILK lines unchanged where they should remain. "
                "No trailing ghost letters or erased old strokes."
            ),
        )
    if "_luna_" in key:
        if target == "heart":
            return (
                "luna,Luna",
                target_tokens,
                (
                    "The neon sign should show one clean heart symbol made from the same red/orange neon tube material. "
                    "Keep the red sign frame, street view, perspective, and glow."
                ),
            )
        return (
            "luna,Luna",
            target_tokens,
            (
                f'The neon sign text must be exactly "{target}", {spelled}. '
                "For ECCV or ICCV, include both repeated letters. Keep the neon tube material, red sign frame, street view, and glow."
            ),
        )
    if "_sign_" in key:
        return (
            "love,LOVE",
            target_tokens,
            (
                f'The first billboard word must be exactly "{target}", spelled {spelled}. '
                f'The full billboard should read "{target} IS ALL YOU NEED". Keep the billboard plane, lamps, buildings, and perspective.'
            ),
        )
    if "_stop_sticker_" in key:
        return (
            "stop,STOP",
            target_tokens,
            (
                f'The comic sticker word must read exactly "{target}", spelled {spelled}. '
                "Keep the sticker cutline, white border, yellow burst, red comic lettering, and natural shadow. "
                "Do not create a rectangular paint smear or crop the sticker."
            ),
        )
    if "_stop_arrow_" in key:
        return (
            "stop,STOP",
            target_tokens,
            (
                f'The top octagonal red sign must read exactly "{target}", spelled {spelled}. '
                "Keep the lower blue circular arrow sign exactly circular with the same upward arrow. "
                "Do not add extra symbols or change the pole or sky."
            ),
        )
    if "_stop_" in key:
        return (
            "stop,STOP",
            target_tokens,
            (
                f'The octagonal red sign must read exactly "{target}", spelled {spelled}, in plain white letters. '
                "Do not add arrows, symbols, or extra icons. Keep the octagonal sign shape, pole, and field background."
            ),
        )
    raise KeyError(key)


def base_target_prompt(prompt: str) -> str:
    markers = [
        " The edited target word must read exactly ",
        " The edited surface must show exactly ",
        " The van front must show ",
    ]
    cut = len(prompt)
    for marker in markers:
        idx = prompt.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return prompt[:cut]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build FE135 T3 Sam-Flow probe v3 manifest from v2 strict failures.")
    ap.add_argument("--v2-manifest", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v2_correct_nodes_manifest.csv"))
    ap.add_argument("--v2-audit", type=Path, default=Path("fe135_t3_samflow_probe_v2_correct_nodes_review/strict_visual_audit_probe.csv"))
    ap.add_argument("--out", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v3_manifest.csv"))
    args = ap.parse_args()

    manifest_rows = read_csv(args.v2_manifest)
    by_pair = {
        (("sam_flow_sd3_probe_v2_correct_nodes" if row["baseline"] == "sam_flow_sd3" else "sam_flow_flux_probe_v2_correct_nodes"), row["task"]): row
        for row in manifest_rows
    }
    rows: list[dict[str, str]] = []
    for audit in read_csv(args.v2_audit):
        if audit["status"] != "fail":
            continue
        source = by_pair[(audit["method"], audit["key"])]
        source_tokens, target_tokens, prompt_extra = tune(source["task"])
        row = {name: source.get(name, "") for name in FIELDNAMES}
        row.update(
            {
                "status": "pending",
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "unchanged_tokens": "",
                "target_prompt": f"{base_target_prompt(source['target_prompt'])} {prompt_extra}",
                "result_image": "",
                "metadata": "",
                "command": "",
                "matched_conditions": "",
                "failure_reason": "",
                "notes": (
                    "FE135 T3 Sam-Flow probe_v3 remaining-fail prompt/token retry; "
                    "fixed seed10; separate probe, not final candidate; "
                    f"v2_failure_type={audit['failure_type']}; v2_note={audit['note']}"
                ),
            }
        )
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)
    print("rows", len(rows))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["baseline"]] = counts.get(row["baseline"], 0) + 1
    for key in sorted(counts):
        print(key, counts[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

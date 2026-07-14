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


def base_target_prompt(prompt: str) -> str:
    markers = [
        " The edited target word must read exactly ",
        " The edited surface must show exactly ",
        " The van front must show ",
        " The gas station sign must read exactly ",
        " Replace only the ",
        " The neon sign ",
        " The first billboard word ",
        " The comic sticker word ",
        " The top octagonal red sign ",
        " The octagonal red sign ",
    ]
    cut = len(prompt)
    for marker in markers:
        idx = prompt.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return prompt[:cut].strip()


def target_from_key(key: str) -> str:
    return {
        "fe_120_gas_station_3_eccv": "ECCV",
        "fe_121_gas_station_4_food": "FOOD",
        "fe_131_groceries_1_bacon": "BACON",
        "fe_133_groceries_3_cvpr": "CVPR",
        "fe_134_groceries_4_eccv": "ECCV",
        "fe_135_groceries_5_iccv": "ICCV",
        "fe_166_luna_3_iccv": "ICCV",
        "fe_167_luna_4_eccv": "ECCV",
        "fe_224_sign_3_iccv": "ICCV",
        "fe_236_stop_2_iccv": "ICCV",
        "fe_246_stop_sticker_3_iccv": "ICCV",
    }[key]


def spelled(target: str) -> str:
    return " ".join(target)


def token_letters(target: str) -> str:
    return ",".join([target.lower(), target.upper(), *target.lower(), "text", "letters"])


def tune(key: str, failure_type: str, note: str) -> tuple[str, str, str]:
    target = target_from_key(key)
    target_tokens = token_letters(target)
    target_spelled = spelled(target)
    feedback = f"Previous failed because: {note}"

    if "_gas_station_" in key:
        return (
            "cafe,CAFE,c,a,f,e,sign,letters",
            target_tokens,
            (
                f'{feedback} Replace the large red word CAFE with exactly "{target}" ({target_spelled}) on the same white board. '
                f'The final sign must contain the full word "{target}" and no CAFE letters. '
                "Use the same red raised-letter style, perspective, lighting, weathering, canopy, cars, and desert background."
            ),
        )

    if "_groceries_" in key:
        item = "first handwritten list item"
        source = "bread,BREAD,b,r,e,a,d,first,line,handwriting"
        return (
            source,
            ",".join([target.lower(), target.upper(), *target.lower(), "marker", "handwriting", "letters"]),
            (
                f'{feedback} Rewrite only the {item} as "{target}" ({target_spelled}) in the same black marker handwriting. '
                "Fully erase the old BREAD strokes before writing the new word; leave no gray ghost letters, trailing marks, or extra dots. "
                "Keep the paper texture, magnet, hand, EGGS line, and MILK line unchanged."
            ),
        )

    if "_luna_" in key:
        return (
            "luna,Luna,l,u,n,a,neon,letters",
            ",".join([target.lower(), target.upper(), *target.lower(), "neon", "tube", "letters"]),
            (
                f'{feedback} The neon tube letters must spell exactly "{target}" ({target_spelled}); include the repeated C twice. '
                "Make the word from continuous red/orange neon tubes on the same sign face. "
                "Keep the red frame, tube glow, wall, street perspective, and background buildings."
            ),
        )

    if "_sign_" in key:
        return (
            "love,LOVE,l,o,v,e,billboard,letters",
            ",".join([target.lower(), target.upper(), *target.lower(), "billboard", "letters"]),
            (
                f'{feedback} The first billboard word must be exactly "{target}" ({target_spelled}), not ICVW or any other spelling. '
                f'The full message must read "{target} IS ALL YOU NEED". Keep the black bold font, white billboard plane, lamps, buildings, and perspective.'
            ),
        )

    if "_stop_sticker_" in key:
        return (
            "stop,STOP,s,t,o,p,comic,sticker,letters",
            ",".join([target.lower(), target.upper(), *target.lower(), "comic", "sticker", "letters"]),
            (
                f'{feedback} The sticker word must read exactly "{target}!" ({target_spelled}) in the existing red comic lettering style. '
                "Preserve the yellow burst, white cutline, shadow, and white background. "
                "Avoid yellow paint smears, gray old-letter residue, rectangular patches, or floating pasted text."
            ),
        )

    if "_stop_" in key:
        return (
            "stop,STOP,s,t,o,p,sign,letters",
            ",".join([target.lower(), target.upper(), *target.lower(), "plain", "white", "letters"]),
            (
                f'{feedback} The octagonal red sign must contain only the plain white letters "{target}" ({target_spelled}). '
                "Do not add arrows, icons, symbols, or extra marks anywhere on the red sign. "
                "Keep the original octagonal sign, pole, field, and soft background."
            ),
        )

    raise KeyError(key)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build FE135 T3 Sam-Flow probe v4 manifest from v3 strict failures.")
    ap.add_argument("--v3-manifest", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v3_manifest.csv"))
    ap.add_argument("--v3-audit", type=Path, default=Path("fe135_t3_samflow_probe_v3_review/strict_visual_audit_probe.csv"))
    ap.add_argument("--out", type=Path, default=Path("remote_patch/fe135_t3_samflow_probe_v4_manifest.csv"))
    args = ap.parse_args()

    manifest_rows = read_csv(args.v3_manifest)
    by_pair = {
        (("sam_flow_sd3_probe_v3" if row["baseline"] == "sam_flow_sd3" else "sam_flow_flux_probe_v3"), row["task"]): row
        for row in manifest_rows
    }

    rows: list[dict[str, str]] = []
    for audit in read_csv(args.v3_audit):
        if audit["status"] != "fail":
            continue
        source = by_pair[(audit["method"], audit["key"])]
        source_tokens, target_tokens, prompt_extra = tune(source["task"], audit["failure_type"], audit["note"])
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
                    "FE135 T3 Sam-Flow probe_v4 remaining-fail prompt/token retry; "
                    "fixed seed10; separate probe, not final candidate; "
                    f"v3_failure_type={audit['failure_type']}; v3_note={audit['note']}; "
                    f"v3_repair_action={audit['repair_action']}"
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

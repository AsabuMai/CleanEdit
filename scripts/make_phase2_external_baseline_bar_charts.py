from __future__ import annotations

import csv
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REMOTE_ROOT = Path("/cluster/users/grad/2025/25t8103/project")
ROOT = REMOTE_ROOT if REMOTE_ROOT.exists() else Path(r"I:\Downloads\1\2")
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
LOCK_DIR = ROOT / "phase2_lock_2026-06-11"
ASSET_DIR = ROOT / "paper" / "assets" if ROOT == REMOTE_ROOT else LOCK_DIR

TABLE2A = EXP / "table2a_phase2_sd3_common_subset_final.csv"
TABLE2B = EXP / "table2b_phase2_native_context_final.csv"
if not TABLE2A.exists():
    TABLE2A = LOCK_DIR / "table2a_phase2_sd3_common_subset_final.csv"
if not TABLE2B.exists():
    TABLE2B = LOCK_DIR / "table2b_phase2_native_context_final.csv"
OUT_SD3 = ASSET_DIR / "phase2_external_baseline_bars_sd3.png"
OUT_FLUX = ASSET_DIR / "phase2_external_baseline_bars_flux_context.png"

METRICS = [
    ("outside_l1", "Non-edit MAE", "lower"),
    ("bg_psnr", "BG-PSNR", "higher"),
    ("bg_lpips_x100", "BG-LPIPS x100", "lower"),
    ("local_clip_t", "Local CLIP-T", "higher"),
]

COLORS = {
    "FlowEdit-SD3": (196, 83, 68),
    "SplitFlow-SD3": (82, 135, 188),
    "Sam-Flow-SD3": (87, 153, 103),
    "FireFlow-FLUX/context": (198, 137, 65),
    "RF-Solver-Edit-FLUX/context": (92, 128, 180),
    "ReFlex-FLUX/context": (150, 102, 174),
    "Sam-Flow-FLUX/context": (78, 151, 137),
    "DeCE-RF-SD3": (37, 104, 126),
}

TEXT = (36, 58, 74)
TEXT_MUTED = (76, 91, 104)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, fill, fnt, spacing: int = 4) -> None:
    chars = max(8, width // max(7, int(getattr(fnt, "size", 12) * 0.55)))
    y = xy[1]
    for line in textwrap.wrap(text, width=chars):
        draw.text((xy[0], y), line, fill=fill, font=fnt)
        y += int(getattr(fnt, "size", 12) * 1.15) + spacing


def nice_limits(values: list[float], direction: str) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if direction == "higher":
        lo = max(0.0, lo - (hi - lo) * 0.25)
        hi = hi + (hi - lo) * 0.18
    else:
        lo = 0.0
        hi = hi * 1.18 if hi > 0 else 1.0
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: list[dict[str, str]],
    metric: tuple[str, str, str],
    title_font,
    label_font,
    small_font,
) -> None:
    key, title, direction = metric
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=(250, 250, 250), outline=(216, 220, 224), width=2)
    draw.text((x0 + 22, y0 + 18), f"{title} ({direction} is better)", fill=TEXT, font=title_font)

    plot_x0 = x0 + 70
    plot_y0 = y0 + 74
    plot_x1 = x1 - 26
    plot_y1 = y1 - 120
    values = [as_float(row, key) for row in rows]
    lo, hi = nice_limits(values, direction)
    axis_color = (120, 126, 132)
    grid_color = (226, 229, 232)

    for tick in range(5):
        t = tick / 4
        y = int(plot_y1 - t * (plot_y1 - plot_y0))
        value = lo + t * (hi - lo)
        draw.line((plot_x0, y, plot_x1, y), fill=grid_color, width=1)
        draw.text((x0 + 18, y - 8), f"{value:.2f}", fill=TEXT_MUTED, font=small_font)
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=axis_color, width=2)
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=axis_color, width=2)

    n = len(rows)
    slot = (plot_x1 - plot_x0) / max(1, n)
    bar_w = max(26, int(slot * 0.54))
    for idx, row in enumerate(rows):
        label = row["label"]
        value = as_float(row, key)
        cx = int(plot_x0 + slot * (idx + 0.5))
        bar_h = int((value - lo) / (hi - lo) * (plot_y1 - plot_y0))
        bx0 = cx - bar_w // 2
        by0 = plot_y1 - bar_h
        color = COLORS.get(label, (96, 122, 150))
        draw.rounded_rectangle((bx0, by0, bx0 + bar_w, plot_y1), radius=5, fill=color)
        draw.text((cx - 22, by0 - 22), f"{value:.3f}", fill=TEXT, font=small_font)
        draw_wrapped(draw, (int(cx - slot * 0.42), plot_y1 + 14), label.replace("/context", ""), int(slot * 0.86), TEXT, label_font, spacing=1)


def make_chart(rows: list[dict[str, str]], labels: list[str], title: str, note: str, out_path: Path) -> None:
    selected = [row for label in labels for row in rows if row.get("label") == label]
    width, height = 2100, 1350
    image = Image.new("RGB", (width, height), (244, 246, 248))
    draw = ImageDraw.Draw(image)
    title_font = font(42, bold=True)
    subtitle_font = font(23)
    panel_title_font = font(24, bold=True)
    label_font = font(15)
    small_font = font(16)

    draw.text((60, 42), title, fill=TEXT, font=title_font)
    draw.text((62, 96), note, fill=TEXT_MUTED, font=subtitle_font)

    left, top = 55, 150
    gap = 32
    panel_w = (width - left * 2 - gap) // 2
    panel_h = (height - top - 58 - gap) // 2
    boxes = [
        (left, top, left + panel_w, top + panel_h),
        (left + panel_w + gap, top, left + panel_w * 2 + gap, top + panel_h),
        (left, top + panel_h + gap, left + panel_w, top + panel_h * 2 + gap),
        (left + panel_w + gap, top + panel_h + gap, left + panel_w * 2 + gap, top + panel_h * 2 + gap),
    ]
    for box, metric in zip(boxes, METRICS):
        draw_panel(draw, box, selected, metric, panel_title_font, label_font, small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(out_path)


def main() -> int:
    table2a = read_rows(TABLE2A)
    table2b = read_rows(TABLE2B)
    dece = [row for row in table2a if row.get("label") == "DeCE-RF-SD3"]
    make_chart(
        table2a,
        ["FlowEdit-SD3", "SplitFlow-SD3", "Sam-Flow-SD3", "DeCE-RF-SD3"],
        "Same-backbone SD3 external baseline comparison",
        "Bars show 15 Phase2 tasks x 3 seeds. FlowAlign is excluded from the paper-facing comparison.",
        OUT_SD3,
    )
    make_chart(
        [*table2b, *dece],
        ["FireFlow-FLUX/context", "RF-Solver-Edit-FLUX/context", "ReFlex-FLUX/context", "Sam-Flow-FLUX/context", "DeCE-RF-SD3"],
        "Native/context external baseline comparison",
        "DeCE-RF-SD3 is shown as a preservation reference; native/context rows are not a strict same-backbone ranking.",
        OUT_FLUX,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

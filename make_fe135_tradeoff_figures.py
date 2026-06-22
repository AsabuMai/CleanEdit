from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJ = Path("/cluster/users/grad/2025/25t8103/project")
EXP = PROJ / "experiments/flowedit135_fixedmask_metrics_20260621"

DISPLAY = {
    "ours_sd3": "Ours",
    "sam_flow_sd3": "SAM-Flow SD3",
    "sam_flow_flux": "SAM-Flow FLUX",
    "flowedit_sd3": "FlowEdit SD3",
    "flowedit_flux": "FlowEdit FLUX",
    "splitflow_sd3": "SplitFlow SD3",
    "otrf_enh_sd3": "OT-RF enh.",
    "drfs_sd3": "DRFS",
    "fireflow": "FireFlow",
    "rf_solver_edit": "RF-Solver",
    "reflex": "ReFLEx",
    "instruct_pix2pix": "IP2P",
    "ledits_pp": "LEDITS++",
}

COLORS = {
    "ours_sd3": "#e66a1f",
    "sam_flow_sd3": "#4c78a8",
    "sam_flow_flux": "#72b7b2",
    "flowedit_sd3": "#59a14f",
    "flowedit_flux": "#8cd17d",
    "splitflow_sd3": "#b279a2",
    "otrf_enh_sd3": "#f2cf5b",
    "drfs_sd3": "#9d755d",
    "fireflow": "#f58518",
    "rf_solver_edit": "#bab0ac",
    "reflex": "#ff9da6",
    "instruct_pix2pix": "#79706e",
    "ledits_pp": "#d37295",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_AXIS = font(24)
F_SMALL = font(18)
F_LABEL = font(19)
F_BOLD = font(21, True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def point(row: dict[str, str]) -> tuple[float, float]:
    edit = float(row["local_clip_t"])
    preserve = -math.log10(max(float(row["bg_lpips"]), 1e-8))
    return edit, preserve


def pareto(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        x, y = point(row)
        dominated = False
        for other in rows:
            if other is row:
                continue
            ox, oy = point(other)
            if ox >= x and oy >= y and (ox > x or oy > y):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda r: point(r)[0])


class Plot:
    def __init__(self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xs: list[float], ys: list[float]):
        self.draw = draw
        self.box = box
        x0, y0, x1, y1 = box
        xpad = max(0.004, (max(xs) - min(xs)) * 0.10)
        ypad = max(0.12, (max(ys) - min(ys)) * 0.10)
        self.xmin = min(xs) - xpad
        self.xmax = max(xs) + xpad
        self.ymin = min(ys) - ypad
        self.ymax = max(ys) + ypad

    def xy(self, x: float, y: float) -> tuple[int, int]:
        x0, y0, x1, y1 = self.box
        px = x0 + (x - self.xmin) / (self.xmax - self.xmin) * (x1 - x0)
        py = y1 - (y - self.ymin) / (self.ymax - self.ymin) * (y1 - y0)
        return int(round(px)), int(round(py))

    def axes(self, xlabel: str = "", ylabel: str = "", title: str | None = None) -> None:
        x0, y0, x1, y1 = self.box
        self.draw.rectangle([x0, y0, x1, y1], fill=(252, 252, 252), outline=(35, 35, 35), width=2)
        for i in range(1, 5):
            gx = x0 + i * (x1 - x0) / 5
            gy = y0 + i * (y1 - y0) / 5
            self.draw.line([(gx, y0), (gx, y1)], fill=(230, 230, 230), width=1)
            self.draw.line([(x0, gy), (x1, gy)], fill=(230, 230, 230), width=1)
        for i in range(6):
            tx = self.xmin + i * (self.xmax - self.xmin) / 5
            ty = self.ymin + i * (self.ymax - self.ymin) / 5
            px, _ = self.xy(tx, self.ymin)
            _, py = self.xy(self.xmin, ty)
            self.draw.text((px - 26, y1 + 10), f"{tx:.3f}", fill=(50, 50, 50), font=F_SMALL)
            self.draw.text((x0 - 58, py - 10), f"{ty:.2f}", fill=(50, 50, 50), font=F_SMALL)
        self.draw.text(((x0 + x1) // 2 - 210, y1 + 48), xlabel, fill=(20, 20, 20), font=F_AXIS)
        self.draw.text((x0 - 70, y0 - 38), ylabel, fill=(20, 20, 20), font=F_AXIS)
        if title:
            self.draw.text((x0, y0 - 52), title, fill=(20, 20, 20), font=F_BOLD)

    def circle(self, x: float, y: float, color: str, r: int = 14, outline=(255, 255, 255), width: int = 3) -> tuple[int, int]:
        px, py = self.xy(x, y)
        self.draw.ellipse([px - r, py - r, px + r, py + r], fill=hex_rgb(color), outline=outline, width=width)
        return px, py

    def star(self, x: float, y: float, color: str, r_outer: int = 25, r_inner: int = 11) -> tuple[int, int]:
        px, py = self.xy(x, y)
        pts = []
        for i in range(10):
            angle = -math.pi / 2 + i * math.pi / 5
            r = r_outer if i % 2 == 0 else r_inner
            pts.append((px + r * math.cos(angle), py + r * math.sin(angle)))
        self.draw.polygon(pts, fill=hex_rgb(color), outline=(0, 0, 0))
        return px, py

    def dashed_line(self, points: list[tuple[float, float]], fill=(140, 140, 140)) -> None:
        pix = [self.xy(x, y) for x, y in points]
        for a, b in zip(pix, pix[1:]):
            x0, y0 = a
            x1, y1 = b
            dist = max(1, math.hypot(x1 - x0, y1 - y0))
            steps = int(dist // 14)
            for i in range(steps + 1):
                if i % 2 == 0:
                    t0 = i / max(1, steps + 1)
                    t1 = min(1.0, (i + 0.7) / max(1, steps + 1))
                    p0 = (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0)
                    p1 = (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)
                    self.draw.line([p0, p1], fill=fill, width=3)


def save_pdf(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path)


def make_overall() -> None:
    rows = read_rows(EXP / "summary_by_method.csv")
    xs, ys = zip(*(point(row) for row in rows))
    image = Image.new("RGB", (1680, 1040), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 42), "Edit-Preservation Trade-off on FlowEdit-135", fill=(20, 20, 20), font=F_TITLE)
    draw.text((80, 86), "Top-right is ideal: stronger localized edit and lower non-edit-region drift.", fill=(80, 80, 80), font=F_AXIS)
    plot = Plot(draw, (140, 170, 1180, 890), list(xs), list(ys))
    plot.axes(
        xlabel="Edit strength: local CLIP-T (higher is better)",
        ylabel="Background preservation: -log10(BG LPIPS)",
    )
    frontier = pareto(rows)
    plot.dashed_line([point(row) for row in frontier])
    for row in rows:
        method = row["method"]
        x, y = point(row)
        if method == "ours_sd3":
            px, py = plot.star(x, y, COLORS[method])
            draw.text((px + 18, py - 35), "Ours", fill=hex_rgb("#b84712"), font=F_BOLD)
            draw.text((px + 18, py - 10), "lowest BG LPIPS", fill=hex_rgb("#b84712"), font=F_SMALL)
        else:
            px, py = plot.circle(x, y, COLORS.get(method, "#999999"), r=15)
            if method in {"sam_flow_sd3", "splitflow_sd3", "flowedit_sd3", "sam_flow_flux", "otrf_enh_sd3", "fireflow"}:
                draw.text((px + 12, py - 12), DISPLAY[method], fill=(35, 35, 35), font=F_LABEL)

    legend_x = 1230
    legend_y = 180
    draw.text((legend_x, legend_y), "Methods", fill=(20, 20, 20), font=F_BOLD)
    for i, row in enumerate(sorted(rows, key=lambda r: DISPLAY.get(r["method"], r["method"]))):
        y = legend_y + 42 + i * 34
        method = row["method"]
        if method == "ours_sd3":
            draw.polygon([(legend_x + 8, y - 12), (legend_x + 14, y), (legend_x + 28, y), (legend_x + 17, y + 8), (legend_x + 22, y + 22), (legend_x + 8, y + 13), (legend_x - 6, y + 22), (legend_x - 1, y + 8), (legend_x - 12, y), (legend_x + 2, y)], fill=hex_rgb(COLORS[method]), outline=(0, 0, 0))
        else:
            draw.ellipse([legend_x - 4, y - 4, legend_x + 20, y + 20], fill=hex_rgb(COLORS.get(method, "#999999")), outline=(255, 255, 255), width=2)
        draw.text((legend_x + 38, y - 5), DISPLAY.get(method, method), fill=(40, 40, 40), font=F_SMALL)
    out = EXP / "tradeoff_overall_edit_preservation.png"
    image.save(out)
    save_pdf(image, EXP / "tradeoff_overall_edit_preservation.pdf")


def make_by_family() -> None:
    rows = read_rows(EXP / "summary_by_family_method.csv")
    families = sorted({row["family_label"] for row in rows})
    image = Image.new("RGB", (2200, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 32), "Ours occupies the high-preservation region across T1-T5", fill=(20, 20, 20), font=F_TITLE)
    draw.text((60, 78), "Grey dots are baselines; orange stars are ours. X: local CLIP-T, Y: -log10(BG LPIPS).", fill=(80, 80, 80), font=F_AXIS)
    panel_w = 400
    gap = 25
    for i, family in enumerate(families):
        fam_rows = [row for row in rows if row["family_label"] == family]
        xs, ys = zip(*(point(row) for row in fam_rows))
        left = 70 + i * (panel_w + gap)
        plot = Plot(draw, (left, 165, left + panel_w, 535), list(xs), list(ys))
        plot.axes(title=family.replace("_", " "))
        for row in fam_rows:
            method = row["method"]
            x, y = point(row)
            if method == "ours_sd3":
                plot.star(x, y, COLORS[method], r_outer=20, r_inner=8)
            else:
                plot.circle(x, y, "#b0b0b0", r=9, outline=(255, 255, 255), width=2)
    out = EXP / "tradeoff_by_family_edit_preservation.png"
    image.save(out)
    save_pdf(image, EXP / "tradeoff_by_family_edit_preservation.pdf")


def make_compact_overall() -> None:
    rows = read_rows(EXP / "summary_by_method.csv")
    xs, ys = zip(*(point(row) for row in rows))
    image = Image.new("RGB", (1320, 860), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 34), "Edit-Preservation Trade-off", fill=(20, 20, 20), font=font(36, True))
    draw.text((70, 80), "FlowEdit-135, 13 methods, 135 edits each", fill=(90, 90, 90), font=font(22))
    plot = Plot(draw, (115, 145, 1000, 705), list(xs), list(ys))
    plot.axes(
        xlabel="Edit strength: local CLIP-T (higher is better)",
        ylabel="Background preservation",
    )
    plot.dashed_line([point(row) for row in pareto(rows)], fill=(130, 130, 130))
    label_methods = {
        "ours_sd3",
        "sam_flow_sd3",
        "splitflow_sd3",
        "flowedit_sd3",
        "sam_flow_flux",
        "otrf_enh_sd3",
        "fireflow",
        "drfs_sd3",
    }
    offsets = {
        "ours_sd3": (22, -42),
        "sam_flow_sd3": (10, -4),
        "splitflow_sd3": (8, 8),
        "flowedit_sd3": (-70, -8),
        "sam_flow_flux": (8, -28),
        "otrf_enh_sd3": (8, -10),
        "fireflow": (8, 4),
        "drfs_sd3": (8, 0),
    }
    for row in rows:
        method = row["method"]
        x, y = point(row)
        if method == "ours_sd3":
            px, py = plot.star(x, y, COLORS[method], r_outer=30, r_inner=13)
            dx, dy = offsets[method]
            draw.text((px + dx, py + dy), "Ours", fill=hex_rgb("#b84712"), font=font(25, True))
            draw.text((px + dx, py + dy + 30), "best preservation", fill=hex_rgb("#b84712"), font=font(18))
        elif method in label_methods:
            px, py = plot.circle(x, y, COLORS.get(method, "#999999"), r=16)
            dx, dy = offsets[method]
            draw.text((px + dx, py + dy), DISPLAY.get(method, method), fill=(35, 35, 35), font=font(17))
        else:
            plot.circle(x, y, "#b8b8b8", r=11, outline=(255, 255, 255), width=2)

    # Direction cues.
    draw.line([(720, 742), (1000, 742)], fill=(45, 45, 45), width=3)
    draw.polygon([(1000, 742), (982, 732), (982, 752)], fill=(45, 45, 45))
    draw.text((730, 755), "stronger edit", fill=(45, 45, 45), font=font(18))
    draw.line([(1040, 650), (1040, 430)], fill=(45, 45, 45), width=3)
    draw.polygon([(1040, 430), (1030, 448), (1050, 448)], fill=(45, 45, 45))
    draw.text((1056, 510), "less background drift", fill=(45, 45, 45), font=font(18))

    draw.text((1070, 160), "Takeaway", fill=(20, 20, 20), font=font(24, True))
    draw.text((1070, 202), "Ours is not the most", fill=(55, 55, 55), font=font(18))
    draw.text((1070, 228), "aggressive editor,", fill=(55, 55, 55), font=font(18))
    draw.text((1070, 254), "but it achieves the", fill=(55, 55, 55), font=font(18))
    draw.text((1070, 280), "lowest non-edit drift", fill=hex_rgb("#b84712"), font=font(18, True))
    draw.text((1070, 306), "while staying on the", fill=(55, 55, 55), font=font(18))
    draw.text((1070, 332), "Pareto frontier.", fill=(55, 55, 55), font=font(18))
    out = EXP / "tradeoff_overall_edit_preservation_compact.png"
    image.save(out)
    save_pdf(image, EXP / "tradeoff_overall_edit_preservation_compact.pdf")


def main() -> None:
    make_overall()
    make_by_family()
    make_compact_overall()
    print(EXP / "tradeoff_overall_edit_preservation.png")
    print(EXP / "tradeoff_by_family_edit_preservation.png")
    print(EXP / "tradeoff_overall_edit_preservation_compact.png")


if __name__ == "__main__":
    main()

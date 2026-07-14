import os, json
from pathlib import Path
from PIL import Image, ImageDraw
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
man = {e["key"]: e for e in json.load(open(PROJ / "data/flowedit_compatible_118/manifest.json"))}
KEYS = os.environ["KEYS"].split(",")
BASE = "outputs/flowedit118_metric_runs"
DECE = "outputs/fe118_r3full_metric_runs"
COLS = [
    ("source", None),
    ("flowedit_sd3", (BASE, "flowedit_sd3")),
    ("sam_flow_sd3", (BASE, "sam_flow_sd3")),
    ("sam_flow_flux", (BASE, "sam_flow_flux")),
    ("CleanEdit-SD3", (DECE, "dece_rf_sd3")),
    ("CleanEdit-FLUX", (DECE, "dece_rf_flux")),
]
S = 320
def L(p):
    try:
        return Image.open(p).convert("RGB").resize((S, S))
    except Exception:
        return Image.new("RGB", (S, S), (35, 35, 35))
rows = []
for k in KEYS:
    e = man[k]
    imgs = [L(PROJ / e["image"])]
    for label, spec in COLS[1:]:
        root, method = spec
        imgs.append(L(PROJ / root / k / method / "seed_10/result.png"))
    cap = man[k]["target_prompt"]
    cap = (cap[:70] + "...") if len(cap) > 70 else cap
    rows.append((k + "  ->  " + cap, imgs))
W = S * len(COLS)
H = (S + 20) * len(rows) + 20
c = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(c)
for j, (label, _) in enumerate(COLS):
    d.text((j * S + 5, 5), label, fill=(0, 0, 0))
y = 20
for lab, imgs in rows:
    d.text((5, y - 1), lab, fill=(170, 0, 0))
    for j, im in enumerate(imgs):
        c.paste(im, (j * S, y + 9))
    y += S + 20
out = PROJ / "tmp/review_shortlist.png"
c.save(out)
print("saved", out, c.size, "rows", len(rows))

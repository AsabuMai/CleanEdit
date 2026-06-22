import os, json, collections
from pathlib import Path
from PIL import Image, ImageDraw
PROJ = Path("/cluster/users/grad/2025/25t8103/project")
man = json.load(open(PROJ / "data/flowedit_compatible_118/manifest.json"))
FAM = os.environ.get("FAM", "T4")
START = int(os.environ.get("START", "0"))
COUNT = int(os.environ.get("COUNT", "100"))
cases = [e for e in man if (e.get("family_label") or "").startswith(FAM)][START:START + COUNT]
# columns: label -> (root, subdir) ; root None means source image
BASE = "outputs/flowedit118_metric_runs"
DECE = "outputs/fe118_r3full_metric_runs"
COLS = [
    ("source", None),
    ("flowedit_sd3", BASE),
    ("sam_flow_sd3", BASE),
    ("sam_flow_flux", BASE),
    ("DeCE-SD3", DECE + "::dece_rf_sd3"),
    ("DeCE-FLUX", DECE + "::dece_rf_flux"),
]
S = 220
def L(p):
    try:
        return Image.open(p).convert("RGB").resize((S, S))
    except Exception:
        return Image.new("RGB", (S, S), (35, 35, 35))
def cell(e, spec):
    if spec is None:
        return L(PROJ / e["image"])
    if "::" in spec:
        root, method = spec.split("::")
    else:
        root, method = spec, spec.rsplit("/", 1)[-1]
        method = None
    key = e["key"]
    if method is None:
        # baseline: root/key/<label-as-method>/seed_10
        return None
    return L(PROJ / root / key / method / "seed_10/result.png")
rows = []
for e in cases:
    imgs = []
    for label, spec in COLS:
        if spec is None:
            imgs.append(L(PROJ / e["image"]))
        elif "::" in spec:
            root, method = spec.split("::")
            imgs.append(L(PROJ / root / e["key"] / method / "seed_10/result.png"))
        else:
            imgs.append(L(PROJ / spec / e["key"] / label / "seed_10/result.png"))
    rows.append((e["family_label"][:2] + " " + e["key"], imgs))
W = S * len(COLS)
H = (S + 18) * len(rows) + 18
c = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(c)
for j, (label, _) in enumerate(COLS):
    d.text((j * S + 4, 4), label, fill=(0, 0, 0))
y = 18
for lab, imgs in rows:
    d.text((4, y - 1), lab, fill=(170, 0, 0))
    for j, im in enumerate(imgs):
        c.paste(im, (j * S, y + 8))
    y += S + 18
out = PROJ / ("tmp/review_%s_%d.png" % (FAM, START))
c.save(out)
print("saved", out, c.size, "rows", len(rows))

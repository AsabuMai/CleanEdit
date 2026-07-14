import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

proj = Path(__file__).resolve().parents[1]
manifest = json.loads((proj / "data/flowedit_compatible_135/manifest_t4_recolor_19.json").read_text())
sd3_root = proj / "outputs/fe135_t4_texture_sd3"
flux_root = proj / "outputs/fe135_t4_texture_flux_h100"
out = proj / "tmp/t4_texture_contact_sd3_flux.png"
out.parent.mkdir(parents=True, exist_ok=True)

thumb = 168
label_h = 42
pad = 8
cols = [("Source", None), ("SD3 texture", sd3_root), ("FLUX texture", flux_root)]
font = ImageFont.load_default()

def fit(im):
    im = im.convert("RGB")
    im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (thumb, thumb), "white")
    x = (thumb - im.width) // 2
    y = (thumb - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas

def result_path(root, key):
    if root.name.endswith("sd3"):
        return root / key / "support_v3_controller_rmsgap/seed_10/result.png"
    return root / key / "dece_rf_flux/seed_10/result.png"

rows = len(manifest)
w = pad + len(cols) * (thumb + pad)
h = pad + rows * (thumb + label_h + pad)
sheet = Image.new("RGB", (w, h), "white")
draw = ImageDraw.Draw(sheet)

for ci, (title, _) in enumerate(cols):
    x = pad + ci * (thumb + pad)
    draw.text((x, 2), title, fill=(0, 0, 0), font=font)

for ri, entry in enumerate(manifest):
    key = entry["key"]
    y = pad + ri * (thumb + label_h + pad) + 12
    for ci, (_, root) in enumerate(cols):
        x = pad + ci * (thumb + pad)
        path = Path(entry["image"]) if root is None else result_path(root, key)
        im = fit(Image.open(path))
        sheet.paste(im, (x, y))
    short = key.replace("fe_", "")
    if len(short) > 54:
        short = short[:51] + "..."
    draw.text((pad, y + thumb + 2), short, fill=(0, 0, 0), font=font)
    target = entry.get("target_prompt", "")
    if len(target) > 86:
        target = target[:83] + "..."
    draw.text((pad, y + thumb + 16), target, fill=(70, 70, 70), font=font)

sheet.save(out)
print(out)

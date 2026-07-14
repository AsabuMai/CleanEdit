import json, os
from PIL import Image, ImageDraw
base = "outputs/phase2_flux_paper_final_20260618"
outdir = "experiments/support_v3_2026-06-02/visual_audit/dece_rf_flux_final_20260618"
os.makedirs(outdir, exist_ok=True)
tasks = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))])
cell = 360
pad = 8
labelh = 24
for seed in ("10","11","12"):
    rows = []
    for t in tasks:
        rundir = os.path.join(base, t, "dece_rf_flux", f"seed_{seed}")
        res = os.path.join(rundir, "result.png")
        meta = os.path.join(rundir, "metadata.json")
        src = None
        if os.path.exists(meta):
            m = json.load(open(meta))
            for k in ("source_image","source","src_image","image"):
                if m.get(k):
                    src = m[k]; break
        imgs = []
        for p in (src, res):
            if p and os.path.exists(p):
                im = Image.open(p).convert("RGB").resize((cell,cell))
            else:
                im = Image.new("RGB",(cell,cell),(40,40,40))
            imgs.append(im)
        row = Image.new("RGB",(cell*2+pad, cell+labelh),(255,255,255))
        row.paste(imgs[0],(0,labelh)); row.paste(imgs[1],(cell+pad,labelh))
        ImageDraw.Draw(row).text((4,6), f"{t}   [source | CleanEdit-FLUX]", fill=(0,0,0))
        rows.append(row)
    W = cell*2+pad
    H = sum(r.height for r in rows)
    canvas = Image.new("RGB",(W,H),(255,255,255))
    y=0
    for r in rows:
        canvas.paste(r,(0,y)); y+=r.height
    outp = os.path.join(outdir, f"dece_rf_flux_final_seed{seed}_grid.png")
    canvas.save(outp)
    print("saved", outp, canvas.size)

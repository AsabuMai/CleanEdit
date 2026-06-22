import json, os
from PIL import Image, ImageDraw
base = "outputs/phase2_flux_paper_final_20260618"
tasks = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))])
seed = "10"
sub = 210          # each sub-image
gap = 4
labelh = 18
cellw = sub*2+gap
cellh = sub+labelh
cols = 3
rows = (len(tasks)+cols-1)//cols
canvas = Image.new("RGB",(cellw*cols + (cols-1)*10, cellh*rows + (rows-1)*10),(255,255,255))
short = {
 "bowl_apple_inside":"apple-in-bowl","brown_bowl_lemon_phase2":"lemon-in-bowl",
 "cat_crown":"cat crown","dog_bow_tie_phase2":"dog bowtie",
 "dog_front_sunglasses_phase2":"dog sunglasses","green_mug_orange_phase2":"mug->orange",
 "mug_heart":"mug heart","pillow_same_color_cable_knit":"knit pillow(white)",
 "pillow_same_color_cable_knit_armchair":"knit pillow(armchair)","pillow_same_color_cable_knit_grey":"knit pillow(grey)",
 "red_office_chair_to_blue_office_chair":"chair red->blue","tote_leaf":"tote leaf",
 "tshirt_star":"tshirt star","white_bowl_orange_tabletop_phase2":"orange on table",
 "yellow_vase_blue_phase2":"vase yellow->blue"}
for i,t in enumerate(tasks):
    r,c = divmod(i,cols)
    rundir = os.path.join(base,t,"dece_rf_flux",f"seed_{seed}")
    res = os.path.join(rundir,"result.png"); meta=os.path.join(rundir,"metadata.json")
    src=None
    if os.path.exists(meta):
        m=json.load(open(meta))
        for k in ("source_image","source","src_image","image"):
            if m.get(k): src=m[k]; break
    cell=Image.new("RGB",(cellw,cellh),(255,255,255))
    for j,p in enumerate((src,res)):
        im = Image.open(p).convert("RGB").resize((sub,sub)) if (p and os.path.exists(p)) else Image.new("RGB",(sub,sub),(40,40,40))
        cell.paste(im,(j*(sub+gap),labelh))
    ImageDraw.Draw(cell).text((3,4), short.get(t,t), fill=(0,0,0))
    canvas.paste(cell,(c*(cellw+10), r*(cellh+10)))
canvas.save("/tmp/flux_compact_seed10.png")
print("saved", canvas.size)

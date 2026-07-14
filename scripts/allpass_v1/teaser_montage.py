import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
P=Path('/cluster/users/grad/2025/25t8103/project')
MR=P/'outputs/flowedit135_metric_runs'
m={e['key']:e for e in json.load(open(P/'data/flowedit_compatible_135/manifest_sam_135.json'))}
ROWS=[('fe_055_cat_7_wooden_sculpture','T5 material: wooden sculpture'),
      ('fe_028_butterfly_1_orange_butterfly','T4 recolor: orange butterfly'),
      ('fe_031_cake_1_berries','T2 insertion: berries'),
      ('fe_094_dog_6_red_top_hat','T1 accessory: red top hat'),
      ('fe_196_penguins_1_origami','T5 material: origami')]
COLS=[('Source',None),('SAM-Flow-SD3','sam_flow_sd3'),('FlowEdit-SD3','flowedit_sd3'),('FlowEdit-FLUX','flowedit_flux'),('OT-RF-SD3','otrf_enh_sd3'),('Ours-SD3','ours_sd3'),('Ours-FLUX','ours_flux')]
T=252;PAD=6;CAP=26;LAB=18
fb=ImageFont.truetype('/usr/share/fonts/google-droid/DroidSans-Bold.ttf',15)
fn=ImageFont.truetype('/usr/share/fonts/google-droid/DroidSans.ttf',13)
def L(meth,e):
    p = P/e['image'] if meth is None else MR/e['key']/meth/'seed_10/result.png'
    try:return Image.open(p).convert('RGB').resize((T,T))
    except:im=Image.new('RGB',(T,T),(50,50,50));ImageDraw.Draw(im).text((8,T//2),'NA',font=fb,fill=(255,80,80));return im
W=len(COLS)*T+(len(COLS)+1)*PAD
H=LAB+len(ROWS)*(CAP+T+PAD)+PAD
img=Image.new('RGB',(W,H),(255,255,255));d=ImageDraw.Draw(img)
for j,(name,_) in enumerate(COLS):
    x=PAD+j*(T+PAD);d.text((x+T//2-30,2),name,font=fb,fill=(0,0,0))
y=LAB
for key,desc in ROWS:
    e=m[key];d.text((PAD,y+2),desc,font=fn,fill=(0,0,0));y+=CAP
    for j,(name,meth) in enumerate(COLS):
        x=PAD+j*(T+PAD);img.paste(L(meth,e),(x,y))
    y+=T+PAD
img.save(P/'tmp_teaser_v2.png');print('saved',img.size)

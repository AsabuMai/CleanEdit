import json, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
P=Path('/cluster/users/grad/2025/25t8103/project')
SD3=P/'outputs/fe135_subjectpreserve_full_v10_sd3'
FLUX=P/'outputs/fe135_full_dece_flux_v11c_pcie8_h100'
OUT=P/'tmp_review_genuine_v10_v11c'; OUT.mkdir(exist_ok=True)
m=json.load(open(P/'data/flowedit_compatible_135/manifest_sam_135.json'))
m=sorted(m,key=lambda e:e['key'])
T=256; PAD=8; CAP=46; COLS=3; PERPAGE=9
fnt=ImageFont.truetype('/usr/share/fonts/google-droid/DroidSans.ttf',13)
fntb=ImageFont.truetype('/usr/share/fonts/google-droid/DroidSans-Bold.ttf',14)
def load(p):
    try: return Image.open(p).convert('RGB').resize((T,T))
    except:
        im=Image.new('RGB',(T,T),(40,40,40)); ImageDraw.Draw(im).text((10,T//2),'MISSING',font=fnt,fill=(255,80,80)); return im
def rowblock(e):
    key=e['key']; fam=e.get('family_label','')
    src=load(P/e['image']); s3=load(SD3/key/'support_v3_controller_rmsgap/seed_10/result.png'); fx=load(FLUX/key/'dece_rf_flux/seed_10/result.png')
    w=COLS*T+(COLS+1)*PAD; h=CAP+T+PAD
    blk=Image.new('RGB',(w,h),(255,255,255)); d=ImageDraw.Draw(blk)
    cap=f'{key}  [{fam}]  ->  {e["target_prompt"]}'
    cap=textwrap.fill(cap,width=150)[:300]
    d.text((PAD,4),cap,font=fnt,fill=(0,0,0))
    for i,(im,lab) in enumerate([(src,'SOURCE'),(s3,'SD3'),(fx,'FLUX')]):
        x=PAD+i*(T+PAD); blk.paste(im,(x,CAP))
        d.rectangle([x,CAP,x+T,CAP+16],fill=(0,0,0)); d.text((x+3,CAP+1),lab,font=fntb,fill=(255,255,0))
    return blk
pages=[m[i:i+PERPAGE] for i in range(0,len(m),PERPAGE)]
for pi,grp in enumerate(pages,1):
    blocks=[rowblock(e) for e in grp]
    w=max(b.width for b in blocks); h=sum(b.height for b in blocks)+PAD
    pg=Image.new('RGB',(w,h),(230,230,230)); y=0
    for b in blocks: pg.paste(b,(0,y)); y+=b.height+PAD
    pg.save(OUT/f'review_p{pi:02d}.png')
print('wrote',len(pages),'pages to',OUT)

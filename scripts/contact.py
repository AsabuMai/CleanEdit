from PIL import Image, ImageDraw
import os
d="data/source_expansion_25/sources"
files=sorted(os.listdir(d))
cell=240; lh=16; cols=4
rows=(len(files)+cols-1)//cols
canvas=Image.new("RGB",(cols*(cell+6), rows*(cell+lh+6)),(255,255,255))
for i,fn in enumerate(files):
    r,c=divmod(i,cols)
    try:
        im=Image.open(os.path.join(d,fn)).convert("RGB").resize((cell,cell))
    except Exception as e:
        im=Image.new("RGB",(cell,cell),(60,0,0))
    tile=Image.new("RGB",(cell,cell+lh),(255,255,255)); tile.paste(im,(0,lh))
    ImageDraw.Draw(tile).text((2,3),fn.replace('web_','').replace('.jpg',''),fill=(0,0,0))
    canvas.paste(tile,(c*(cell+6), r*(cell+lh+6)))
canvas.save("/tmp/new_sources_contact.png"); print("saved",canvas.size,"n",len(files))

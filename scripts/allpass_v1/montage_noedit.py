import json,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
P=Path('/cluster/users/grad/2025/25t8103/project')
SD3=P/'outputs/fe135_subjectpreserve_full_v10_sd3'; FLUX=P/'outputs/fe135_full_dece_flux_v11c_pcie8_h100'
OUT=P/'tmp_noedit_review'; OUT.mkdir(exist_ok=True)
m={e['key']:e for e in json.load(open(P/'data/flowedit_compatible_135/manifest_sam_135.json'))}
KEYS=['fe_246_stop_sticker_3_iccv','fe_224_sign_3_iccv','fe_248_this_must_be_the_place_2_cvpr','fe_142_iguana_2_blue_lizard_top_hat','fe_186_muffins_1_strawberries','fe_036_cake_red_blueberries_2_raspberries','fe_140_horse_4_brown_horse','fe_010_beer_glass_1_cocktail','fe_207_pizza_tomato_olive_1_pepperoni','fe_067_cat_stone_1_black_kitten','fe_242_stop_arrow_5_home','fe_019_brown_owl_1_white_owl','fe_197_penguins_2_sculpture','fe_004_bear_grass_1_black_bear','fe_205_pizza_slice_1_pepperoni','fe_221_rocks_6_colorful_wooden_blocks','fe_027_butterflies_1_yellow','fe_241_stop_arrow_4_love','fe_243_stop_arrow_6_beer','fe_199_piece_of_cake_2_red_velvet_cake','fe_194_parrots_3_sculpture','fe_000_bear_1_black_bear','fe_028_butterfly_1_orange_butterfly','fe_146_japanese_castle_1_lego_castle']
T=320;PAD=8;CAP=40;PER=5
fnt=ImageFont.truetype('/usr/share/fonts/google-droid/DroidSans.ttf',14);fb=ImageFont.truetype('/usr/share/fonts/google-droid/DroidSans-Bold.ttf',15)
def L(p):
    try:return Image.open(p).convert('RGB').resize((T,T))
    except:
        im=Image.new('RGB',(T,T),(40,40,40));ImageDraw.Draw(im).text((8,T//2),'NONE',font=fb,fill=(255,80,80));return im
def row(k):
    e=m[k];src=L(P/e['image']);s3=L(SD3/k/'support_v3_controller_rmsgap/seed_10/result.png');fx=L(FLUX/k/'dece_rf_flux/seed_10/result.png')
    w=3*T+4*PAD;h=CAP+T+PAD;b=Image.new('RGB',(w,h),(255,255,255));d=ImageDraw.Draw(b)
    d.text((PAD,3),textwrap.fill(k+' -> '+e['target_prompt'],160)[:200],font=fnt,fill=(0,0,0))
    for i,(im,lab) in enumerate([(src,'SOURCE'),(s3,'SD3'),(fx,'FLUX')]):
        x=PAD+i*(T+PAD);b.paste(im,(x,CAP));d.rectangle([x,CAP,x+74,CAP+17],fill=(0,0,0));d.text((x+3,CAP+1),lab,font=fb,fill=(255,255,0))
    return b
pages=[KEYS[i:i+PER] for i in range(0,len(KEYS),PER)]
for pi,g in enumerate(pages,1):
    bs=[row(k) for k in g];w=max(b.width for b in bs);h=sum(b.height for b in bs)+PAD
    pg=Image.new('RGB',(w,h),(225,225,225));y=0
    for b in bs:pg.paste(b,(0,y));y+=b.height+PAD
    pg.save(OUT/f'noedit_p{pi}.png')
print('wrote',len(pages),'pages')

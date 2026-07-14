import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
P=Path('/cluster/users/grad/2025/25t8103/project'); MR=P/'outputs/flowedit135_metric_runs'
m={e['key']:e for e in json.load(open(P/'data/flowedit_compatible_135/manifest_sam_135.json'))}
COLS=[('Source',None),('Ours-SD3','ours_sd3'),('Ours-FLUX','ours_flux'),('SAM-Flow-SD3','sam_flow_sd3'),('SplitFlow-SD3','splitflow_sd3'),('FlowEdit-SD3','flowedit_sd3'),('OT-RF-SD3','otrf_enh_sd3'),('DRFS-SD3','drfs_sd3'),('SAM-Flow-FLUX','sam_flow_flux'),('FlowEdit-FLUX','flowedit_flux'),('FireFlow','fireflow'),('RF-Solver','rf_solver_edit'),('ReFLEx','reflex'),('IP2P','instruct_pix2pix'),('LEDITS++','ledits_pp')]
FAM={'T1_attached_accessory':['fe_046_cat_crown_1_black_top_hat','fe_094_dog_6_red_top_hat','fe_095_dog_7_jeweled_crown','fe_192_parrots_1_top_hat','fe_195_parrots2_1_crown'],
'T2_container_insertion':['fe_031_cake_1_berries','fe_180_milk_4_whipped_cream','fe_201_pizza_1_pineapple_ham','fe_198_piece_of_cake_1_cherry_on_top','fe_010_beer_glass_1_cocktail'],
'T3_surface_decal':['fe_118_gas_station_1_cvpr','fe_222_sign_1_cvpr','fe_235_stop_1_cvpr','fe_164_luna_1_sol','fe_116_free_wifi_1_free_beer'],
'T4_local_recolor':['fe_000_bear_1_black_bear','fe_128_gray_bird_2_red_bird','fe_108_duck_1_colorful_duck','fe_140_horse_4_brown_horse','fe_015_bikes_5_green_bicycle'],
'T5_same_color_material':['fe_055_cat_7_wooden_sculpture','fe_003_bear_4_sculpture','fe_080_corgi_1_lego_bricks','fe_139_horse_3_bronze_sculpture','fe_196_penguins_1_origami']}
T=168;PAD=3;CAP=16
fb=ImageFont.truetype('/usr/share/fonts/google-droid/DroidSans-Bold.ttf',12)
def L(meth,e):
    p=P/e['image'] if meth is None else MR/e['key']/meth/'seed_10/result.png'
    try:return Image.open(p).convert('RGB').resize((T,T))
    except:im=Image.new('RGB',(T,T),(50,50,50));ImageDraw.Draw(im).text((6,T//2),'NA',font=fb,fill=(255,90,90));return im
for fam,keys in FAM.items():
    keys=[k for k in keys if k in m]
    W=len(COLS)*T+(len(COLS)+1)*PAD; H=CAP+len(keys)*(T+PAD)+PAD
    img=Image.new('RGB',(W,H),(255,255,255)); d=ImageDraw.Draw(img)
    for j,(name,_) in enumerate(COLS):
        x=PAD+j*(T+PAD); d.text((x+2,2),name,font=fb,fill=(0,0,0))
    y=CAP
    for k in keys:
        e=m[k]
        for j,(name,meth) in enumerate(COLS):
            x=PAD+j*(T+PAD); img.paste(L(meth,e),(x,y))
        y+=T+PAD
    img.save(P/('tmp_appendix_'+fam+'.png')); print('wrote',fam,img.size)

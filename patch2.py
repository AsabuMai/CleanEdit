import shutil
f = "flux/flux_hrec.py"
shutil.copy(f, f + ".bak6_20260618")
s = open(f, encoding="utf-8").read()

# --- MUG green branch: drop MaxFilter dilation, re-gate by source-green to kill bg/slab spill ---
mug_old = '''                mug_img = mug_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
                mug_alpha = np.asarray(mug_img).astype(np.float32) / 255.0
                shade = np.clip((src_luma / 0.36) ** 1.10, 0.38, 1.35)'''
mug_new = '''                mug_img = mug_img.filter(ImageFilter.GaussianBlur(0.5))
                mug_alpha = np.asarray(mug_img).astype(np.float32) / 255.0
                gate = ((g > r) & (chroma > 0.03)).astype(np.float32)
                gate = np.asarray(Image.fromarray((gate * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(0.6))).astype(np.float32) / 255.0
                mug_alpha = mug_alpha * gate
                shade = np.clip((src_luma / 0.36) ** 1.10, 0.38, 1.35)'''
assert s.count(mug_old) == 1, f"mug match={s.count(mug_old)}"
s = s.replace(mug_old, mug_new)

# --- VASE recolor_full_mask branch: tighten outward expansion ---
vase_old = '''                            comp_img = Image.fromarray((np.maximum(component, reference_mask.astype(np.float32)) * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5))'''
vase_new = '''                            comp_img = Image.fromarray((np.maximum(component, reference_mask.astype(np.float32)) * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.7))'''
assert s.count(vase_old) == 1, f"vase match={s.count(vase_old)}"
s = s.replace(vase_old, vase_new)

open(f, "w", encoding="utf-8").write(s)
print("flux_hrec.py patched OK (backup .bak6_20260618)")

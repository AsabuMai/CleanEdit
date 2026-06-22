import shutil, io
f = "flux/flux_hrec.py"
shutil.copy(f, f + ".bak5_20260618")
src = open(f, encoding="utf-8").read()
old = '''                mug_img = mug_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.0))
                mug_alpha = np.asarray(mug_img).astype(np.float32) / 255.0
                target_shaded = np.clip(target * np.clip(src_luma / 0.36, 0.45, 1.15), 0.0, 1.0)
                alpha = np.clip(mug_alpha[..., None] * float(args.final_recolor_blend_scale), 0.0, 1.0)
                result = source_arr * (1.0 - alpha) + target_shaded * alpha'''
new = '''                mug_img = mug_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
                mug_alpha = np.asarray(mug_img).astype(np.float32) / 255.0
                shade = np.clip((src_luma / 0.36) ** 1.10, 0.38, 1.35)
                target_shaded = np.clip(target * shade, 0.0, 1.0)
                gen_luma = result.mean(axis=2, keepdims=True)
                gen_blur = np.asarray(Image.fromarray((np.clip(gen_luma[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(4.0))).astype(np.float32) / 255.0
                detail = np.clip(gen_luma[..., 0] - gen_blur, -0.25, 0.35)[..., None]
                target_shaded = np.clip(target_shaded + 0.6 * detail, 0.0, 1.0)
                alpha = np.clip(mug_alpha[..., None] * float(args.final_recolor_blend_scale), 0.0, 1.0)
                result = source_arr * (1.0 - alpha) + target_shaded * alpha'''
assert src.count(old) == 1, f"expected 1 match, got {src.count(old)}"
open(f, "w", encoding="utf-8").write(src.replace(old, new))
print("patched OK; backup .bak5_20260618")

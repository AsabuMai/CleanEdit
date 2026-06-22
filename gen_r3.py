import json, collections
PROJ = "/cluster/users/grad/2025/25t8103/project"
man = json.load(open(PROJ + "/data/flowedit_compatible_118/manifest_118_run.json"))
byfam = collections.defaultdict(list)
for e in man: byfam[e["family_label"]].append(e)
take = {"T1_attached_accessory":3,"T2_container_insertion":3,"T4_local_recolor":3,"T5_same_color_material":2}
probe = []
for fam,n in take.items(): probe += byfam[fam][:n]
json.dump(probe, open(PROJ+"/data/flowedit_compatible_118/manifest_probe2.json","w"), indent=1)
json.dump({"add":{"pbud":0.65,"pgain":0.8},"recolor":{"edit_hedit":0.60,"local_target":0.40,"edit_region":0.28,"pbud":0.30,"pgain":1.5}}, open(PROJ+"/r3_flux_kind.json","w"))
json.dump({"add":{"--adaptive-edit-gain":"3.5"},"recolor":{"--adaptive-edit-gain":"3.0"}}, open(PROJ+"/r3_flux_adapt.json","w"))
json.dump({"recolor":{"hedit":"0.40","color":"0.20"}}, open(PROJ+"/r3_sd3_kind.json","w"))
json.dump({"add":{"--adaptive-preserve-drift-budget":"0.30","--adaptive-preserve-gain":"2.0","--adaptive-edit-gain":"3.5","--adaptive-edit-target-rms":"0.58"},"recolor":{"--adaptive-preserve-drift-budget":"0.30","--adaptive-preserve-gain":"2.5","--adaptive-edit-gain":"3.0","--adaptive-edit-target-rms":"0.55"}}, open(PROJ+"/r3_sd3_adapt.json","w"))
print("probe2 cases", len(probe))
for e in probe: print(" ", e["family_label"][:2], e["key"])

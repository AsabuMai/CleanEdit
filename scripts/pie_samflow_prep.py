import json, urllib.request, ssl, time, re
from pathlib import Path
ctx=ssl.create_default_context()
PROJ=Path("/cluster/users/grad/2025/25t8103/project")
man=json.load(open(PROJ/"data/pie_pilot_20260618/manifest.json"))
def fetch_row(cfg, idx):
    url=f"https://datasets-server.huggingface.co/rows?dataset=UB-CVML-Group/PIE_Bench_pp&config={cfg}&split=V1&offset={idx}&length=1"
    for _ in range(4):
        try: return json.load(urllib.request.urlopen(url,timeout=40,context=ctx))["rows"][0]["row"]
        except Exception as ex: print("retry",ex); time.sleep(3)
    raise RuntimeError(cfg)
toks={}
for e in man:
    r=fetch_row(e["config"], e["row_idx"])
    bw=json.loads(r["blended_words"]) if r.get("blended_words") else []
    src=[]; tgt=[]
    for pair in bw:
        p=pair.split(","); a=p[0].strip(); b=p[1].strip() if len(p)>1 else ""
        if a: src.append(a)
        if b: tgt.append(b)
    raw_tgt=r["target_prompt"]
    if not tgt: tgt=[w.strip() for seg in re.findall(r"\[([^\]]+)\]", raw_tgt) for w in [seg]]
    if not src: src=[r["source_prompt"].strip().split()[-1]]
    # dedup keep order, cap 2
    def cap(xs):
        out=[]
        for x in xs:
            if x and x not in out: out.append(x)
        return out[:2]
    toks[e["key"]]={"source":cap(src),"target":cap(tgt)}
    print(f"{e['key']:12s} src={toks[e['key']]['source']} tgt={toks[e['key']]['target']}  ({r['source_prompt'][:30]} -> {raw_tgt[:34]})")
json.dump(toks, open(PROJ/"data/pie_pilot_20260618/samflow_tokens.json","w"), indent=2)
print("wrote samflow_tokens.json n=",len(toks))

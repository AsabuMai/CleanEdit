import json, re
from pathlib import Path
P=Path('/cluster/users/grad/2025/25t8103/project')
reg=json.load(open(P/'data/flowedit_compatible_135/repair_allpass_v1/registry.json'))
m={x['key']:x for x in json.load(open(P/'data/flowedit_compatible_135/manifest_sam_135.json'))}
c9=sorted(k for k,v in reg.items() if v['category'].startswith('C9'))
pat=re.compile(r'^A[n]?\s+(?:(bronze|golden|gold|marble|wooden|wood|sand|stone|silver|granite|clay)\s+)?(statue|sculpture)\s+of\s+(?:a\s+|an\s+)?(.+)$', re.I|re.S)
rew={}
for k in c9:
    tgt=m[k]['target_prompt']
    mt=pat.match(tgt.strip())
    if not mt:
        print('NO MATCH:',k); continue
    mat=(mt.group(1) or 'stone').lower(); rest=mt.group(3).strip()
    matphrase = f'{mat} material' if mat!='stone' else 'carved sculpted stone'
    new=f'A {rest} The whole body and skin are turned into {matphrase}, while the pose, body silhouette, face position, camera view, and background stay exactly the same.'
    rew[k]={'orig':tgt,'new':new,'negative':'a different person, new identity, changed pose, different body shape, statue pedestal, base, plinth, second statue, duplicated figure'}
json.dump(rew,open(P/'data/flowedit_compatible_135/repair_allpass_v1/c9_repair_prompts.json','w'),indent=2,ensure_ascii=False)
print('total C9 rewritten:',len(rew))
# build 2-case smoke manifest
SMOKE=['fe_148_jump_kick_2_bronze_statue','fe_171_meditation_1_wooden_statue']
sm=[]
for k in SMOKE:
    e=dict(m[k]); e['target_prompt']=rew[k]['new']; e['negative_prompt']=rew[k]['negative']; sm.append(e)
    print('SMOKE',k,'->',rew[k]['new'][:90])
json.dump(sm,open(P/'data/flowedit_compatible_135/manifest_c9_smoke2.json','w'),indent=2,ensure_ascii=False)
print('wrote manifest_c9_smoke2.json')

import json,math
from t import load
for d in load('g1_reach_heights_heldout'):
    i=d['spec']['id'][-5:]
    if i not in ('s5001','s5008','s5010','s5015','s5020','s5009'): continue
    a=d['agent'];h=d['truth']['handle_center'];tp=a['target']['position']
    rp=d['reference']['posture']
    print(i,'ref stance',rp['stance'],rp['arm'],'handle true',[round(x,3) for x in h],'perc',tp)
    print(' agent fixture est',a['target'].get('fixture'),' obstacles',a['target']['obstacles'])
    print(' true fixture',d['truth']['fixture']['center'],d['truth']['fixture']['half'],d['truth']['fixture']['yaw'])
    for p in a['plans'][:1]:
        fl=p['failures']
        print(' nfail listed',len(fl),'of',p['failed'])
        near=[f for f in fl if math.hypot(f['stance'][0]-rp['stance'][0],f['stance'][1]-rp['stance'][1])<0.12]
        print(' failures near ref stance',near)

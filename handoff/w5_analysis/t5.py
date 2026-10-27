import json,math,sys
from t import load
import numpy as np
def fixerr(d):
    a=d['agent'];t=d['truth']['fixture']
    f=(a.get('target') or {}).get('fixture')
    if not f:return None
    top_e=(f['center'][2]+f['half_extents'][2])-(t['center'][2]+t['half'][2])
    # yaw mod pi/2 aware of half swap: compare footprint via corner support in x direction toward robot
    dy=(f['yaw']-t['yaw']+math.pi/2)%math.pi-math.pi/2
    dc=math.hypot(f['center'][0]-t['center'][0],f['center'][1]-t['center'][1])
    return round(top_e,2),round(dc,2),round(math.degrees(dy)),[round(x,2) for x in f['half_extents'][:2]],[round(x,2) for x in t['half'][:2]]
for name in ('g1_reach_heights_heldout',):
    for d in load(name):
        e=d['evaluator']
        if d['agent']['verdict']=='refused' or e['reference_feasible']:
            print(d['spec']['id'][-5:],'refF' if e['reference_feasible'] else 'refI',d['agent']['verdict'],'fix(top_err,ctr_err,yaw_err_deg,est_half,true_half)=',fixerr(d))
print('--- executed episodes: stops, execution, stance achieved vs planned')
for name in sys.argv[1:]:
    for d in load(name):
        a=d['agent'];e=d['evaluator'];ex=a.get('execution') or {}
        if not ex: continue
        if e['physical_success'] and e['claimed_success']: continue
        print(name[-12:],d['spec']['id'][-9:],a['verdict'],'|',a['reason'][:60],'| stops',[(round(s[0],1),s[1]) if isinstance(s,list) else s for s in (ex.get('stops') or [])][:2],'| ps',[(round(p['t'],1),p['reason']) for p in e['protective_stops']][:2],'| peakTq',(d.get('execution_trace') or {}).get('peak_torque_ratio'),'| corr',[(c['stage'][:3],round(c['error'],3),c['action'][:4]) for c in ex.get('correction',[])][:8],'| mind',e['min_distance_during_m'])

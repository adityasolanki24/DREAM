import math,numpy as np,json
from t import load
AR=(0.35,0.4,0.45,0.5,0.6)
print('HELDOUT ref-feasible, non-success: reference chosen stance vs agent grid; plus agent plan+roll counts')
for d in load('g1_reach_heights_heldout'):
    e=d['evaluator']
    if not e['reference_feasible']: continue
    rp=d['reference']['posture'];h=d['truth']['handle_center']
    dx,dy=rp['stance'][0]-h[0],rp['stance'][1]-h[1]
    r=math.hypot(dx,dy);ang=math.degrees(math.atan2(dy,dx))%360
    off=min(ang%45,45-ang%45)
    onr=min(abs(r-x) for x in AR)
    a=d['agent']
    print(d['spec']['id'][-5:],a['verdict'],'ok' if e['physical_success'] else '--','ref r=%.2f ang=%.0f grid_off_deg=%.0f r_off=%.2f'%(r,ang,off,onr),'pelv',rp['pelvis_height'],'pitch',rp['torso_rpy'][1],'arm',rp['arm'],'refrolls',len(d['reference']['rollouts']),'agent_cands',[p['feasible'] for p in a['plans']])
# rollout hand_error: accepted vs rejected, agent, heldout + dev
import collections
for name in ('g1_reach_heights_heldout','g1_reach_heights_dev'):
    acc=[];rej=[];rejh=[];reasons=collections.Counter();tq=[]
    for d in load(name):
        for r in d['agent']['rollouts']:
            m=r['metrics'];he=m.get('hand_error')
            if r['passed']:acc.append(he)
            else:
                reasons[tuple(r['reasons'])]+=1
                if not any(s.startswith('stop') or s.startswith('refused') for s in r['reasons']): rej.append(he)
            tq.append((m.get('peak_torque_ratio'),r['passed']))
    q=lambda v:[round(float(x),3) for x in np.percentile(v,[0,25,50,75,100])] if v else None
    print(name,'accepted n',len(acc),q(acc),'| rejected-by-gate(no stop) n',len(rej),q(rej),'| gate hand_error in (0.04,0.06):',sum(1 for x in rej if x and 0.04<x<=0.06))
    print(' reasons',reasons.most_common(12))

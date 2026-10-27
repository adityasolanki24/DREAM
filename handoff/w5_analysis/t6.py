import math,numpy as np
from t import load
def sd(c,h,yaw,p):
    cs,sn=math.cos(yaw),math.sin(yaw)
    x,y=p[0]-c[0],p[1]-c[1]
    lx,ly=cs*x+sn*y,-sn*x+cs*y
    dx,dy=abs(lx)-h[0],abs(ly)-h[1]
    return math.hypot(max(dx,0),max(dy,0))+min(max(dx,dy),0)
for d in load('g1_reach_heights_heldout'):
    a=d['agent'];t=d['truth'];f=(a.get('target') or {}).get('fixture')
    if not f: continue
    hc=t['handle_center']; tf=t['fixture']
    print(d['spec']['id'][-5:],d['agent']['verdict'][:4],'refF' if d['evaluator']['reference_feasible'] else 'refI',
      'handle->true box surface %.3f'%sd(tf['center'],tf['half'],tf['yaw'],hc),' ->est box surface %.3f'%sd(f['center'],f['half_extents'],f['yaw'],hc),
      'est face stand-off: handle in est box?' )

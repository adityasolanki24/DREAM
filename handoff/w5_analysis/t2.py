import json,glob,math,sys
from t import load,n
def cls(d):
    a=d['agent'];e=d['evaluator'];ref=d['reference']
    v=a['verdict'];ok=e['physical_success']
    if ok and e['claimed_success']: return 'SUCCESS',''
    h=d['truth']['handle_center']
    rr=a['rollouts']
    if not e['reference_feasible'] and v=='refused': return 'ok-refuse',''
    if ok: return 'C',f"fingertip {e['true_distance_m']*1000:.1f}mm true, perceived {a.get('measured_error',0)*1000:.1f}, thr {a['claim_threshold']*1000:.1f}"
    r=a['reason']
    if 'not found' in r: return 'P','not found'
    if 'handle lost' in r: return 'P/N','handle lost after walk; '+('rollouts:'+str([x['reasons'] for x in rr]))
    if v=='refused':
        pl=a['plans']; feas=sum(p['feasible'] for p in pl)
        if feas==0 or 'rollout' not in r:
            if rr and not r.startswith("no feasible plan within the measured envelope: {'rollout'"): return 'K/R?',r[:80]
            return 'E/K',"planner 0 feasible: "+r[60:160]
        return 'R',r[:100]
    return 'S/other',r[:100]
for d in sys.argv[1:]:
    for x in load(d):
        c,i=cls(x);e=x['evaluator'];a=x['agent'];ref=x['reference']
        h=x['truth']['handle_center'];rp=ref.get('posture');rad=None
        if rp: rad=round(math.hypot(rp['stance'][0]-h[0],rp['stance'][1]-h[1]),2)
        nref=None
        for k,r in enumerate(ref.get('rollouts',[])):
            if r['passed']: nref=k+1
        tg=a.get('target');pe=round(math.dist(tg['position'],h)*1000,1) if tg else None
        print(x['spec']['id'][-6:],'h=%.2f'%x['spec']['height'],a['verdict'],'refF' if e['reference_feasible'] else 'refI','->',c,'|perr_mm',pe,'|refrad',rad,'ref_roll_to_pass',nref,'/',len(ref.get('rollouts',[])),'|',i)

import json,glob,sys,math
R='C:/Users/24adi/DREAM-wholebody/results/'
def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(R+d+'/agent/*.json'))]
def n(x):return round(x,3) if isinstance(x,float) else x
def row(d):
    a=d['agent'];e=d['evaluator'];sp=d['spec'];t=d['truth']
    tg=a.get('target') or {}
    perr=None
    if tg.get('position'):
        perr=math.dist(tg['position'],t['handle_center'])
    ref=d.get('reference',{})
    rr=a['rollouts']
    ex=a.get('execution') or {}
    plans=a['plans']
    rrej=[(r['stance'][:2],r['arm'],r['reasons'],n(r['metrics'].get('hand_error')),n(r['metrics'].get('peak_torque_ratio')),n(r['metrics'].get('min_com_margin'))) for r in rr]
    return dict(id=sp['id'][-5:],h=n(sp['height']),dist=n(sp['distance']),verdict=a['verdict'],ok=e['physical_success'],claimed=e['claimed_success'],reff=e['reference_feasible'],
      perr=n(perr),sig=tg.get('sigma'),views=tg.get('views'),truedist=n(e['true_distance_m']),mind=e.get('min_distance_during_m'),
      reason=a['reason'][:230],plans=[(p['stage'],p['feasible'],p['failed']) for p in plans],
      nroll=len(rr),rollout=rrej[:6],stops=ex.get('stops'),corr=[(c['stage'],n(c['error']),c['action']) for c in ex.get('correction',[])],
      ps=e['protective_stops'],ref_plans=ref.get('n_plans'),ref_post=ref.get('posture'),
      reftop=[(r['stance'],r['arm'],r['passed'],r['reasons'],n(r.get('hand_error'))) for r in ref.get('rollouts',[])][:4],
      post=a.get('posture') and (a['posture']['stance'],a['posture']['arm'],a['posture']['pelvis_height']),
      mt=a.get('measured_error'),ct=a.get('claim_threshold'),restances=a.get('restances'),trace=d.get('execution_trace') and {k:n(d['execution_trace'].get(k)) for k in('peak_torque_ratio','peak_tilt','min_com_margin')})
if __name__=='__main__':
    for d in sys.argv[1:]:
        for x in load(d):
            print(json.dumps(row(x),default=str));print()

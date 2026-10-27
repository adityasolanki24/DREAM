import sys,math,json,pathlib
sys.path.insert(0,'C:/Users/24adi/DREAM-wholebody')
import numpy as np
from t import load
from dream.robot.g1_posture import PlannerConfig,PosturePlanner,PerceivedBox
from dream.robot.wholebody import ControllerCapabilities
from dream.sim.reach_scenes import sample_reach_scene
from dream.eval.reach import true_boxes
import dataclasses
ROOT=pathlib.Path('C:/Users/24adi/DREAM-wholebody')
cfg=json.load(open(ROOT/'experiments/configs/g1_reach_heights.json'))
caps=ControllerCapabilities.from_envelope_v2(ROOT/cfg['envelope'])
pc=cfg['planner']
agentcfg=PlannerConfig(radii=tuple(pc['radii']),n_angles=pc['n_angles'],max_stances=pc['max_stances'])
dense=dataclasses.replace(agentcfg,radii=(0.35,0.4,0.45,0.5,0.6,0.7),n_angles=16,max_stances=10000)
def normal(fx,p):
    c,h,y=fx['center'],fx['half_extents'],fx['yaw']
    cs,sn=math.cos(y),math.sin(y);x,yy=p[0]-c[0],p[1]-c[1]
    l=np.array([cs*x+sn*yy,-sn*x+cs*yy]);s=l/np.array(h[:2])
    i=int(np.argmax(abs(s)));n=np.zeros(2);n[i]=np.sign(l[i])
    w=np.array([[cs,-sn],[sn,cs]])@n
    return (-w[0],-w[1],0.0)
ids=sys.argv[1:]
for d in load('g1_reach_heights_heldout'):
    i=d['spec']['id'][-5:]
    if i not in ids: continue
    sp=d['spec'];a=d['agent'];tg=a['target']
    scene=sample_reach_scene(sp['seed'],sp['height'],sp['distance'],sp['lateral'],sp['yaw_jitter'],sp['n_crates'],sp['second_fixture'])
    pb=[PerceivedBox(tuple(o['center']),tuple(o['half_extents']),o['yaw'],bool(o['hand_may_touch'])) for o in tg['obstacles']]
    tb=true_boxes(scene)
    ptar=np.array(tg['position']);ttar=np.asarray(scene.handle_center)
    cov=np.array(tg['covariance']);ax=normal(tg['fixture'],ptar)
    pose=(0,0,0)
    def run(name,cf,tar,boxes,cv,axis):
        pl=PosturePlanner(caps,cf)
        r=pl.plan(tar,cv,pose,boxes,approach_axis=axis)
        print(i,name,'feasible',len(r.plans),'failed',len(r.failures),r.reasons())
    run('A agent-as-run      ',agentcfg,ptar,pb,cov,ax)
    run('B -approach_axis    ',agentcfg,ptar,pb,cov,None)
    run('C -axis,-cov        ',agentcfg,ptar,pb,None,None)
    run('D true boxes+tgt,axis,cov',agentcfg,ttar,tb,cov,ax)
    run('E true boxes+tgt,none',agentcfg,ttar,tb,None,None)
    run('F E but dense grid  ',dense,ttar,tb,None,None)
    run('G perceived, dense,none',dense,ptar,pb,None,None)

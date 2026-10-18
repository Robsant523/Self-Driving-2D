import subprocess
import sys
import torch
from src.common import ROOT, config
from src.sim.env import BatchedDrivingEnv
from src.sim.render import CrashHistory, Renderer
from src.tracks import DEFAULT_TRACK, apply_track


def test_crash_history_keeps_pre_reset_pose_and_ignores_timeouts():
    cfg=config(ROOT/'configs/bendy.yaml');cfg.update(device='cpu',num_envs=2)
    env=BatchedDrivingEnv(cfg)
    env.pos[0,1]=env.road.sample(env.pos[:1,0])[0]+cfg['width']/2
    crashed_pos=env.pos[:1].clone();heading=env.heading[:1].clone()
    env.steps[1]=cfg['max_steps']-1
    _,_,_,trunc,info=env.step(torch.zeros(2,2))
    assert info['collision'].tolist()==[True,False] and trunc[1]
    assert not torch.equal(crashed_pos,env.pos[:1])
    history=CrashHistory(2);history.capture(info)
    # Another step occurs before the next rendered frame.
    info2=env.step(torch.zeros(2,2))[-1];history.capture(info2);history.drain()
    assert len(history.cars[0])==1 and len(history.cars[1])==0
    torch.testing.assert_close(torch.tensor(history.cars[0][0][:2]),crashed_pos[0])
    assert history.cars[0][0][2]==heading[0]
    history.drain();assert len(history.cars[0])==1


def test_history_survives_camera_replacement_and_limit():
    cfg=config(ROOT/'configs/bendy.yaml');cfg.update(device='cpu',num_envs=1)
    env=BatchedDrivingEnv(cfg);history=CrashHistory(1,limit=2)
    for x in (10.,20.,30.):
        env.pos[0]=torch.tensor([x,env.road.sample(torch.tensor([x]))[0].item()+cfg['width']/2])
        info=env.step(torch.zeros(1,2))[-1];history.capture(info)
    renderer=Renderer(1,show_rays=True,history=history)
    renderer.frame(env);renderer.close()
    assert len(history.cars[0])==2 and history.cars[0][0][0]==20.
    replacement=Renderer(1,camera='overview',history=history)
    try:
        replacement.frame(env)
        assert len(replacement.history.cars[0])==2
        # Ghost patches are faded and stay underneath the active car.
        assert any(p.get_alpha()==.38 for p in replacement.axes[0].patches)
    finally:replacement.close()


def test_road_preview_arrows_match_policy_lookahead():
    cfg=apply_track(config(ROOT/'configs/bendy.yaml'),DEFAULT_TRACK);cfg.update(device='cpu',num_envs=1)
    env=BatchedDrivingEnv(cfg);renderer=Renderer(1,show_rays=True)
    try:
        renderer.frame(env)
        from matplotlib.text import Annotation
        arrows=[a for a in renderer.axes[0].texts if isinstance(a,Annotation)]
        assert len(arrows)==cfg['rays']+3
        for arrow,distance in zip(arrows[-3:],(5.,12.,20.)):
            target_x=env.pos[0,0].item()+distance
            target_y=env.road.sample(torch.tensor([target_x]))[0].item()
            assert abs(arrow.xy[0]-target_x)<1e-5 and abs(arrow.xy[1]-target_y)<1e-5
    finally:renderer.close()


def test_default_training_uses_long_course_and_sufficient_horizon(tmp_path):
    result=subprocess.run([sys.executable,'-m','src.train','--device','cpu','--num-envs','2','--updates','1','--output',str(tmp_path)],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    checkpoint=torch.load(tmp_path/'checkpoint.pt',weights_only=True)
    assert checkpoint['config']['road']['name']=='Long Winding'
    assert checkpoint['config']['length']==600
    assert checkpoint['config']['max_steps']>=1600
    assert checkpoint['model']['body.0.weight'].shape[1]==25

import copy
import torch
import pytest
import yaml
from src.common import ROOT, config
from src.tracks import apply_track
from src.sim.env import BatchedDrivingEnv
from src.sim.render import Renderer
from src.sim.world import collision


@pytest.fixture
def cfg():
    c=config(ROOT/'configs/bendy.yaml');c.update(device='cpu',num_envs=2)
    return c


def load(tmp_path,cfg,track):
    p=tmp_path/'course.yaml';p.write_text(yaml.safe_dump(track))
    return apply_track(cfg,p)


def test_custom_interpolation_obstacles_and_checkpoint_roundtrip(tmp_path,cfg):
    c=load(tmp_path,cfg,{'width':10,'points':[[0,0],[20,0],[50,6]],'obstacles':[[12,0,1]]})
    env=BatchedDrivingEnv(c)
    torch.testing.assert_close(env.road.sample(torch.tensor([10.,35.,60.]))[0],torch.tensor([0.,3.,6.]))
    expected=torch.tensor([[[12.,0.,1.]],[[12.,0.,1.]]])
    torch.testing.assert_close(env.obstacles,expected)
    env.reset(seed=89)
    torch.testing.assert_close(env.obstacles,expected)
    assert env.observe().shape==(2,25)
    env.pos[:]=torch.tensor([2.,0.]);env.heading.zero_()
    from src.sim.raycast import cast
    rays=cast(env.pos,env.heading,env.obstacles,torch.zeros(1),20.)
    torch.testing.assert_close(rays,torch.full((2,1),9.))
    assert collision(torch.tensor([[8.,0.]]),torch.tensor([[13.,0.]]),env.obstacles[:1],c,env.road).item()
    # Config contains all geometry; original YAML may be removed after training.
    torch.save({'config':c},tmp_path/'checkpoint.pt')
    (tmp_path/'course.yaml').unlink()
    restored=BatchedDrivingEnv(torch.load(tmp_path/'checkpoint.pt',weights_only=True)['config'])
    torch.testing.assert_close(restored.obstacles,expected)


@pytest.mark.parametrize('patch,match',[
    ({'points':[[0,0],[0,1],[40,0]]},'increase'),
    ({'points':[[1,0],[40,0]]},'start'),
    ({'points':[[0,0],[20,float('nan')]]},'finite'),
    ({'width':1},'narrow'),
    ({'points':[[0,0],[20,100]]},'steep'),
    ({'obstacles':[[12,0,-1]]},'positive'),
    ({'obstacles':[[2,0,1]]},'after'),
    ({'obstacles':[[12,6,1]]},'inside'),
    ({'obstacles':[[12,0]]},'radius'),
    ({'obstacels':[]},'Unknown'),
])
def test_invalid_tracks_fail_early(tmp_path,cfg,patch,match):
    track={'width':10,'points':[[0,0],[40,0]],'obstacles':[]};track.update(patch)
    with pytest.raises(ValueError,match=match): load(tmp_path,cfg,track)


def test_example_is_valid_and_fixed_on_masked_reset(cfg):
    c=apply_track(cfg,ROOT/'tracks/custom.yaml');env=BatchedDrivingEnv(c)
    before=env.obstacles.clone()
    env.steps[0]=c['max_steps']-1
    _,_,_,trunc,_=env.step(torch.zeros(2,2))
    assert trunc.tolist()==[True,False]
    torch.testing.assert_close(env.obstacles,before)


def test_overview_camera_fits_course_and_stays_fixed(cfg):
    env=BatchedDrivingEnv(apply_track(cfg,ROOT/'tracks/custom.yaml'))
    renderer=Renderer(1,camera='overview')
    try:
        frame=renderer.frame(env,'test')
        ax=renderer.axes[0];xlim,ylim=ax.get_xlim(),ax.get_ylim()
        assert xlim[0]<0 and xlim[1]>env.cfg['length']
        points=env.cfg['road']['points'];half=env.cfg['width']/2
        assert ylim[0]<min(p[1] for p in points)-half
        assert ylim[1]>max(p[1] for p in points)+half
        assert frame.shape[2]==3 and renderer.map_ax is None
        env.pos[:,0]=75;renderer.frame(env,'test')
        assert ax.get_xlim()==xlim and ax.get_ylim()==ylim
    finally: renderer.close()


def test_follow_camera_still_moves(cfg):
    env=BatchedDrivingEnv(cfg);r=Renderer(1,camera='follow')
    try:
        r.frame(env);before=r.axes[0].get_xlim()
        env.pos[:,0]+=10;r.frame(env)
        assert r.axes[0].get_xlim()[0]==before[0]+10
    finally:r.close()


def test_narrow_valid_course_spawns_inside(tmp_path,cfg):
    c=load(tmp_path,cfg,{'width':3.4,'points':[[0,0],[40,0]],'obstacles':[]})
    env=BatchedDrivingEnv(c)
    assert not env.road.collision(env.pos,env.pos,c['car_radius']).any()

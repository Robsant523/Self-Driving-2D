import torch
import pytest
from src.common import config, ROOT
from src.sim.env import BatchedDrivingEnv
from src.sim.road import Road, follow_road


@pytest.fixture
def cfg():
    c = config(ROOT / 'configs/bendy.yaml')
    c.update(device='cpu', num_envs=8)
    return c


def test_flat_road_rays_and_parallel_misses(cfg):
    cfg['road']['amplitude'] = 0
    road = Road(cfg, 'cpu')
    rays = road.rays(torch.tensor([[10.,0.]]),torch.zeros(1),torch.tensor([0.,torch.pi/2,-torch.pi/2]),20.)
    torch.testing.assert_close(rays,torch.tensor([[20.,4.,4.]]))


def test_swept_road_collision_and_tangent_contact(cfg):
    cfg['road']['amplitude'] = 0
    road = Road(cfg, 'cpu')
    start = torch.tensor([[10.,0.],[10.,0.],[10.,0.],[10.,0.]])
    end = torch.tensor([[11.,0.],[11.,8.],[11.,2.5],[11.,2.49]])
    assert road.collision(start,end,1.5).tolist() == [False,True,True,False]


def test_road_spawn_reset_and_determinism(cfg):
    a,b = BatchedDrivingEnv(cfg),BatchedDrivingEnv(cfg)
    assert a.obs_dim == 25
    torch.testing.assert_close(a.heading,a.road.sample(a.pos[:,0])[1])
    assert not a.road.collision(a.pos,a.pos,cfg['car_radius']).any()
    for _ in range(4):
        left,right = a.step(follow_road(a)),b.step(follow_road(b))
        for x,y in zip(left[:4],right[:4]): torch.testing.assert_close(x,y)
    a.steps[0]=cfg['max_steps']-1
    previous=a.steps.clone()
    obs,_,term,trunc,info=a.step(follow_road(a))
    assert trunc[0] and not term[0] and a.steps[0]==0
    torch.testing.assert_close(a.steps[1:],previous[1:]+1)
    torch.testing.assert_close(a.heading[:1],a.road.sample(a.pos[:1,0])[1])
    assert obs.shape==(8,25) and torch.isfinite(obs).all()
    assert info['final_observation'][0,16]>0 and obs[0,16]==0


def test_curve_sensors_see_bend(cfg):
    env=BatchedDrivingEnv(cfg)
    rays=env.road.rays(env.pos,env.heading,torch.zeros(1),100.)
    assert (rays<100).all() and (rays>0).all()


def test_reference_controller_finishes_bends(cfg):
    env=BatchedDrivingEnv(cfg)
    active=torch.ones(env.n,dtype=torch.bool)
    finished=torch.zeros_like(active)
    crashed=torch.zeros_like(active)
    for _ in range(cfg['max_steps']):
        _,_,term,trunc,info=env.step(follow_road(env))
        finished |= active & info['success']
        crashed |= active & info['collision']
        active &= ~(term|trunc)
    assert finished.all() and not crashed.any()


def test_road_obstacles_spawn_on_road(cfg):
    cfg['obstacles']=12
    env=BatchedDrivingEnv(cfg)
    delta=env.obstacles[...,1]-env.road.sample(env.obstacles[...,0])[0]
    assert (delta.abs() <= (cfg['width']-3)/2).all()


def test_corridor_cannot_use_road_controller():
    c=config();c.update(device='cpu',num_envs=1)
    with pytest.raises(ValueError,match='bendy'):
        follow_road(BatchedDrivingEnv(c))

import copy
import pytest
import torch
from src.common import config
from src.sim.env import BatchedDrivingEnv
from src.sim.raycast import cast
from src.sim.vehicle import advance
from src.sim.world import collision
from src.rl.buffer import gae, RolloutBuffer
from src.rl.networks import ActorCritic
from src.rl.ppo import update

@pytest.fixture
def cfg():
    c=config(); c.update(device='cpu',num_envs=4,obstacles=2)
    return c

@pytest.mark.parametrize('circle,expected', [([5.,0.,1.],4.),([-5.,0.,1.],20.),([5.,1.,1.],5.),([0.,0.,1.],0.),([5.,2.,1.],20.),([30.,0.,1.],20.)])
def test_ray(circle,expected):
    result=cast(torch.zeros(1,2),torch.zeros(1),torch.tensor([[circle]]),torch.zeros(1),20.)
    torch.testing.assert_close(result,torch.tensor([[expected]]))

def test_empty_and_wall():
    result=cast(torch.zeros(2,2),torch.zeros(2),torch.empty(2,0,3),torch.tensor([0.,torch.pi/2]),20.,12.)
    torch.testing.assert_close(result,torch.tensor([[20.,6.],[20.,6.]]))

def test_nearest_rotated():
    result=cast(torch.zeros(1,2),torch.tensor([torch.pi/2]),torch.tensor([[[0.,8.,1.],[0.,5.,1.]]]),torch.zeros(1),20.)
    torch.testing.assert_close(result,torch.tensor([[4.]]))

def test_dynamics(cfg):
    pos,heading,speed=advance(torch.zeros(4,2),torch.zeros(4),torch.ones(4)*2,torch.zeros(4,2),cfg)
    torch.testing.assert_close(pos[:,0],torch.ones(4)*.2)
    assert (pos[:,1]==0).all() and (heading==0).all()
    _,turned,_=advance(pos,heading,speed,torch.tensor([[1.,0.]]).repeat(4,1),cfg)
    assert (turned>0).all()

def test_swept_collision(cfg):
    hit=collision(torch.tensor([[0.,0.]]),torch.tensor([[10.,0.]]),torch.tensor([[[5.,0.,.5]]]),cfg)
    assert hit.item()

def test_shapes_and_determinism(cfg):
    a=BatchedDrivingEnv(cfg); b=BatchedDrivingEnv(cfg)
    for _ in range(8):
        action=torch.tensor([[.1,.5]]).repeat(4,1)
        left=a.step(action); right=b.step(action)
        for x,y in zip(left[:4],right[:4]): torch.testing.assert_close(x,y)
        assert left[0].shape==(4,19) and left[0].dtype==torch.float32
        assert left[2].dtype==torch.bool and left[1].shape==(4,)
    with pytest.raises(ValueError): a.step(torch.zeros(2,2))

def test_masked_resets_and_final_observation(cfg):
    cfg['obstacles']=0
    e=BatchedDrivingEnv(cfg)
    e.steps[0]=cfg['max_steps']-1
    old=e.pos.clone()
    obs,_,term,trunc,info=e.step(torch.tensor([[0.,1.]]).repeat(4,1))
    assert trunc.tolist()==[True,False,False,False]
    assert not term.any()
    assert e.steps.tolist()==[0,1,1,1]
    assert obs[0,-3]==0 and info['final_observation'][0,-3]>0
    torch.testing.assert_close(e.pos[1:,0],old[1:,0]+.04)

def test_collision_and_success(cfg):
    cfg['obstacles']=0
    e=BatchedDrivingEnv(cfg)
    e.pos[0,1]=cfg['width']/2
    e.pos[1,0]=cfg['length']+.1
    _,reward,term,trunc,info=e.step(torch.zeros(4,2))
    assert term.tolist()==[True,True,False,False]
    assert info['collision'][0] and info['success'][1]
    assert reward[0]<0 and reward[1]>0
    assert not trunc.any()

def test_gae_bootstrap_and_trace_boundaries():
    r=torch.ones(2,3);v=torch.zeros_like(r);nv=torch.ones_like(r)*10
    term=torch.tensor([[True,False,False],[False,False,False]])
    trunc=torch.tensor([[False,True,False],[False,False,False]])
    adv,_=gae(r,v,nv,term,trunc,.9,1.)
    torch.testing.assert_close(adv[0],torch.tensor([1.,10.,19.]))

def test_policy_and_ppo(cfg):
    torch.manual_seed(0)
    e=BatchedDrivingEnv(cfg);m=ActorCritic(e.obs_dim,32)
    b=RolloutBuffer(4,4,e.obs_dim,e.device)
    obs=e.observe()
    with torch.no_grad():
        for t in range(4):
            b.obs[t]=obs
            action,b.raw[t],b.log_probs[t],b.values[t]=m.act(obs)
            assert (action.abs()<=1).all()
            logp,_,_=m.evaluate(obs,b.raw[t])
            torch.testing.assert_close(logp,b.log_probs[t])
            obs,b.rewards[t],b.terminated[t],b.truncated[t],info=e.step(action)
            b.next_values[t]=m.distribution(info['final_observation'])[1]
    before=m.actor.weight.detach().clone()
    metrics=update(m,torch.optim.Adam(m.parameters(),lr=.001),b,{**cfg['ppo'],'epochs':2,'minibatch':8})
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert not torch.equal(before,m.actor.weight)

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_mps(cfg):
    cfg['device']='mps'
    e=BatchedDrivingEnv(cfg)
    obs=e.step(torch.zeros(4,2,device='mps'))[0]
    assert obs.device.type=='mps' and torch.isfinite(obs).all()

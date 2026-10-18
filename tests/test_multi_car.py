import numpy as np
import pytest
import torch
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.text import Annotation
from src.common import ROOT, config
from src.sim.env import BatchedDrivingEnv
from src.sim.render import CrashHistory, Renderer
from src.tracks import apply_track


def make_env(count=3, *, track=None, obstacles=0):
    cfg=config(ROOT/'configs/bendy.yaml')
    cfg.update(device='cpu',num_envs=count,obstacles=obstacles)
    if track is not None:
        cfg=apply_track(cfg,ROOT/track)
    env=BatchedDrivingEnv(cfg)
    # Distinct actual training poses make missing or duplicated lanes detectable.
    env.pos[:,0]=torch.arange(count,dtype=torch.float32)*20+10
    env.pos[:,1]=env.road.sample(env.pos[:,0])[0]
    env.heading[:]=torch.arange(count,dtype=torch.float32)*.1
    return env


def bodies(ax, *, crashed=False):
    return [patch for patch in ax.patches
            if isinstance(patch,FancyBboxPatch) and (patch.get_alpha()==.38)==crashed]


def body_center(ax, patch):
    return ax.transData.inverted().transform(patch.get_transform().transform((0.,0.)))


@pytest.mark.parametrize('camera',['overview','follow'])
def test_overlay_draws_actual_lanes_in_distinct_colors_and_keeps_all_crashes(camera):
    env=make_env()
    original=env.observe().clone();rng=env.generator.get_state().clone()
    history=CrashHistory(env.n)
    crash_pos=env.pos.clone()+torch.tensor([[-2.,1.],[-3.,1.],[-4.,1.]])
    history.capture({'collision':torch.tensor([True,False,True]),
                     'final_pos':crash_pos,
                     'final_heading':env.heading.clone(),
                     'final_steering':torch.zeros(env.n)})
    renderer=Renderer(env.n,camera=camera,layout='overlay',show_rays=True,history=history)
    try:
        frame=renderer.frame(env)
        assert frame.ndim==3 and frame.shape[2]==3
        assert len(renderer.axes)==1
        ax=renderer.axes[0];active=bodies(ax);ghosts=bodies(ax,crashed=True)
        assert len(active)==env.n and len(ghosts)==2
        assert len({patch.get_facecolor() for patch in active})==env.n
        np.testing.assert_allclose([body_center(ax,p) for p in active],env.pos.numpy(),atol=1e-6)
        np.testing.assert_allclose([body_center(ax,p) for p in ghosts],crash_pos[[0,2]].numpy(),atol=1e-6)
        assert [len(lane) for lane in history.cars]==[1,0,1]
        # Shared-road overlays keep the focus car's sensor and preview arrows legible.
        arrows=[text for text in ax.texts if isinstance(text,Annotation) and text.arrow_patch is not None]
        assert len(arrows)==env.cfg['rays']+3
        # Repainting must neither add duplicate crash markers nor alter learning state.
        renderer.frame(env)
        assert len(bodies(ax,crashed=True))==2
        torch.testing.assert_close(env.observe(),original,rtol=0,atol=0)
        assert torch.equal(env.generator.get_state(),rng)
    finally:
        renderer.close()


def test_overlay_draws_shared_fixed_obstacles_once():
    env=make_env(track='tracks/custom.yaml')
    renderer=Renderer(env.n,camera='overview',layout='overlay')
    try:
        renderer.frame(env)
        obstacles=[patch for patch in renderer.axes[0].patches if isinstance(patch,Circle)]
        assert len(obstacles)==env.cfg['obstacles']
        np.testing.assert_allclose([p.center for p in obstacles],env.obstacles[0,:,:2].numpy())
        assert len(bodies(renderer.axes[0]))==env.n
    finally:
        renderer.close()


def test_overlay_rejects_different_random_obstacle_worlds():
    env=make_env(obstacles=2)
    assert not torch.equal(env.obstacles[0],env.obstacles[1])
    renderer=Renderer(env.n,layout='overlay')
    try:
        with pytest.raises(ValueError,match='(?i)obstacle'):
            renderer.frame(env)
    finally:
        renderer.close()


def test_default_grid_preserves_each_training_world():
    env=make_env(obstacles=1)
    env.obstacles[:,0,:]=torch.tensor([[15.,0.,.5],[35.,1.,.7],[55.,2.,.9]])
    renderer=Renderer(env.n,camera='overview')
    try:
        frame=renderer.frame(env)
        assert frame.shape[2]==3 and renderer.layout=='grid'
        assert sum(ax.get_visible() for ax in renderer.axes)==env.n
        for lane,ax in enumerate(renderer.axes[:env.n]):
            assert len(bodies(ax))==1
            np.testing.assert_allclose(body_center(ax,bodies(ax)[0]),env.pos[lane].numpy(),atol=1e-6)
            obstacles=[patch for patch in ax.patches if isinstance(patch,Circle)]
            assert len(obstacles)==1
            np.testing.assert_allclose(obstacles[0].center,env.obstacles[lane,0,:2].numpy())
            assert obstacles[0].radius==pytest.approx(env.obstacles[lane,0,2].item())
    finally:
        renderer.close()

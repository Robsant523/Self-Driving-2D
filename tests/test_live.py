import base64
import io
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
import torch
from PIL import Image
from src.common import ROOT, config
from src.live import LiveTrainingView
from src.sim.env import BatchedDrivingEnv


def test_live_server_frame_camera_and_readonly_state():
    cfg=config(ROOT/'configs/bendy.yaml');cfg.update(device='cpu',num_envs=2)
    env=BatchedDrivingEnv(cfg)
    original=env.observe().clone();rng=env.generator.get_state().clone()
    view=LiveTrainingView(open_browser=False)
    try:
        assert view.server.server_address[0]=='127.0.0.1'
        view.frame(env,pace=False,phase='Collecting experience',transitions=2)
        with urlopen(view.url+'/state',timeout=3) as response:
            data=json.load(response)
        assert data['phase']=='Collecting experience' and data['transitions']==2
        picture=Image.open(io.BytesIO(base64.b64decode(data['frame'])))
        assert picture.size==(1200,480)
        torch.testing.assert_close(env.observe(),original)
        assert torch.equal(env.generator.get_state(),rng)
        request=Request(view.url+'/camera',data=b'{"camera":"follow"}',headers={'Content-Type':'application/json','Origin':view.url})
        with urlopen(request,timeout=3) as response: assert response.status==200
        view.frame(env,pace=False)
        assert view.renderer.camera=='follow'
        for body,origin,status in [(b'{"camera":"bad"}',view.url,400),(b'{"camera":"follow"}','https://example.com',403)]:
            request=Request(view.url+'/camera',data=body,headers={'Content-Type':'application/json','Origin':origin})
            with pytest.raises(HTTPError) as error:urlopen(request,timeout=3)
            assert error.value.code==status
        with pytest.raises(HTTPError) as error:urlopen(view.url+'/README.md',timeout=3)
        assert error.value.code==404
    finally:
        view.close()
    assert not view.thread.is_alive()


def test_live_training_matches_headless_and_exits(tmp_path):
    common=[sys.executable,'-m','src.train','--track','tracks/presets/gentle_s.yaml',
            '--device','cpu','--num-envs','2','--updates','1']
    for name,extra in [('headless',[]),('live',['--live','--no-browser','--live-exit','--live-every','64','--live-fps','30'])]:
        result=subprocess.run(common+extra+['--output',str(tmp_path/name)],cwd=ROOT,capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stderr
        assert (tmp_path/name/'learning.png').exists()
    a=torch.load(tmp_path/'headless/checkpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'live/checkpoint.pt',weights_only=True)
    assert a['update']==b['update']==1
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)

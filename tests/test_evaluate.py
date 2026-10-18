from src.common import config
from src.evaluate import evaluate


def test_evaluation_counts_exact_first_episodes():
    cfg=config(); cfg.update(device='cpu',obstacles=0,max_steps=3)
    a=evaluate(cfg,episodes=5,seed=90)
    b=evaluate(cfg,episodes=5,seed=90)
    assert a==b
    assert a['episodes']==5 and a['mean_length']==3
    assert a['timeout_rate']==1 and a['collision_rate']==0

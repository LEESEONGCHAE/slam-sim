import numpy as np

from slamsim.control import PIDSteer
from slamsim.envs import PathFollowEnv, pid_policy, run_episode
from slamsim.tracks import Track


def test_oval_curvature_matches_theory():
    t = Track.oval(80, 40)
    assert abs(t.kappa.max() - 80 / 40 ** 2) < 0.002


def test_errors_sign():
    t = Track.oval()
    x, y, h = t.start_pose()
    # 왼쪽으로 1 m 떨어진 점 → e = +1
    e, epsi = t.errors(0, x - np.sin(h), y + np.cos(h), h)
    assert abs(e - 1.0) < 1e-6 and abs(epsi) < 1e-9


def test_pid_ff_follows_track():
    env = PathFollowEnv(randomize=False, episode_time=15.0)
    m, _ = run_episode(env, pid_policy(PIDSteer(feedforward=True)), track=Track.random(100))
    assert not m["off"] and m["rms_e"] < 0.2


def test_zero_steer_leaves_track():
    env = PathFollowEnv(randomize=False, episode_time=15.0)
    m, _ = run_episode(env, lambda o, e: 0.0, track=Track.random(100))
    assert m["off"]


def test_observation_shape():
    env = PathFollowEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()

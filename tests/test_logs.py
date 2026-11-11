import math
import numpy as np

from slamsim import load_carmen, record_sim, SensorLog, ScanMsg, OdomMsg
from slamsim.geometry import compose, relative


def test_compose_relative_roundtrip():
    a, b = (1.0, 2.0, 0.7), (-3.0, 0.5, -2.9)
    c = compose(a, relative(a, b))
    assert np.allclose(c, b)


def test_carmen_parse(tmp_path):
    p = tmp_path / "t.clf"
    ranges = " ".join(["1.5"] * 179 + ["81.83"])
    p.write_text(
        "# comment\n"
        "ODOM 0.0 0.0 0.1 0 0 0 100.0 nohost 0\n"
        f"FLASER 180 {ranges} 0.0 0.0 0.1 0.5 0.2 0.1 100.5 nohost 0.5\n")
    log = load_carmen(str(p))
    assert isinstance(log.messages[0], OdomMsg) and isinstance(log.messages[1], ScanMsg)
    s = log.messages[1]
    assert math.isclose(s.t, 0.5) and s.odom == (0.5, 0.2, 0.1)
    assert s.valid.sum() == 179 and not s.valid[-1]
    assert math.isclose(log.angles[0], -math.pi / 2)


def test_sim_log_save_load(tmp_path):
    log = record_sim(seconds=3, seed=2)
    assert log.truth is not None and len(log.scans) == 30
    f = tmp_path / "s.npz"
    log.save(f)
    log2 = SensorLog.load(f)
    assert len(log2.messages) == len(log.messages)
    assert np.allclose(log2.scans[5].ranges, log.scans[5].ranges)
    assert log2.truth_at(1.0) is not None

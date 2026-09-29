import pytest

from ber import compute


def test_caps_from_config():
    assert 1 <= compute.n_jobs() <= 32
    assert compute.max_ram_gb() == 200
    assert isinstance(compute.use_gpu(), bool) and compute.gpu_device() == compute.gpu_ids()[0] and compute.max_vram_gb() == 40


def test_gpu_first_policy_in_config():
    assert compute.use_gpu() is True and compute.cpu_fallback() is False


def test_pool_runs_and_monitor_logs_peak():
    with compute.RamMonitor("unit", interval=0.05) as mon:
        with compute.pool(2) as p:
            assert p.map(abs, [-1, -2, 3]) == [1, 2, 3]
    assert mon.peak_gb > 0


def test_monitor_raises_over_cap():
    with pytest.raises(compute.RamCapExceeded):
        with compute.RamMonitor("tiny-cap", interval=0.01, cap_gb=1e-6):
            import time
            time.sleep(0.05)


def test_env_can_only_lower_n_jobs(monkeypatch):
    monkeypatch.setenv("BER_N_JOBS", "4")
    assert compute.n_jobs() == 4
    monkeypatch.setenv("BER_N_JOBS", "999")
    assert compute.n_jobs() <= 32

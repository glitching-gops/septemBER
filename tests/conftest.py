"""Test-session settings.

Unit tests run on toy data, for which AGENTS.md allows the CPU code paths. Production code
never falls back to CPU (``compute.cpu_fallback: false``; GPU errors propagate and the run
stops). Tests that run a toy top-k opt in to the fallback with
``pytestmark = pytest.mark.usefixtures("toy_data_cpu_fallback")`` so they do not fail just
because the shared GPUs are busy; the policy test itself still sees ``cpu_fallback() is False``.
"""
import pytest

from ber import compute


@pytest.fixture
def toy_data_cpu_fallback(monkeypatch):
    monkeypatch.setattr(compute, "cpu_fallback", lambda: True)

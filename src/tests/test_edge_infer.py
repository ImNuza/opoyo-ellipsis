from __future__ import annotations

import pytest

from shared.schemas import InferenceResult, SensorWindow
from edge.infer import Classifier, FakeCnn, FusionCnn, JointCnn, MODELS, StubCnn, load_runtime


def _window() -> SensorWindow:
    zeros = [0.0] * 100
    return SensorWindow(
        node_id="Phone 1",
        room=1,
        t_start_ms=1000,
        t_end_ms=3000,
        hz=50.0,
        mag=zeros,
        ax=zeros,
        ay=zeros,
        az=zeros,
        db=[-40.0] * 100,
    )


def test_stub_returns_required_shape():
    result = StubCnn().infer(_window())
    assert isinstance(result, InferenceResult)
    assert result.inference_id
    assert result.timestamp == 3000
    assert result.node_id == "Phone 1"
    assert result.room == 1
    assert result.is_fall is False
    assert result.confidence == 0.0


def test_stub_does_not_raise_on_zero_mag():
    StubCnn().infer(_window())


def test_fake_cnn_is_swappable() -> None:
    clf: Classifier = FakeCnn(is_fall=True, confidence=0.95)
    result = clf.infer(_window())
    assert result.is_fall is True
    assert result.confidence == 0.95
    assert result.timestamp == 3000


@pytest.mark.skipif(not (MODELS / "fuse_head.joblib").exists(), reason="no trained heads")
def test_fusion_zero_mag_without_pcm_does_not_raise():
    result = FusionCnn().infer(_window())
    assert isinstance(result, InferenceResult)
    assert 0.0 <= result.confidence <= 1.0
    assert result.node_id == "Phone 1"
    assert result.timestamp == 3000


@pytest.mark.skipif(not (MODELS / "fuse_head.joblib").exists(), reason="no trained heads")
def test_load_runtime_prefers_the_joint_head():
    clf = load_runtime()
    # JointCnn (one logistic over hand features + embedding) beats the
    # two-stage stack in every condition measured, so load_runtime picks it
    # first and only falls back to FusionCnn if joint_head.joblib is absent.
    assert isinstance(clf, (JointCnn, FusionCnn))

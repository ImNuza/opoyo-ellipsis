from types import SimpleNamespace

import numpy as np

from opoyo.classify import RuleClassifier, load_classifier
from opoyo.config import CFG
from opoyo.features import extract
from opoyo.synth import body_like, click_like


def test_rule_scores_body_high_and_click_low():
    rng = np.random.default_rng(0)
    clf = RuleClassifier()
    fs = 100.0
    body = extract(body_like(fs, rng), fs)
    pan = extract(click_like(fs, rng), fs)
    assert clf.score(body) > 0.6
    assert clf.score(pan) < 0.4


def test_load_classifier_falls_back_when_model_missing(tmp_path):
    cfg = SimpleNamespace(classify=SimpleNamespace(model_path=str(tmp_path / "clf.joblib")))
    clf = load_classifier(cfg)
    assert isinstance(clf, RuleClassifier)


def test_load_classifier_cfg_missing_joblib():
    # models/clf.joblib is the configured path; absent → rule baseline, no raise
    clf = load_classifier(CFG)
    from pathlib import Path
    from opoyo.config import ROOT

    p = Path(CFG.classify.model_path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        assert isinstance(clf, RuleClassifier)

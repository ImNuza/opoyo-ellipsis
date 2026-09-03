from shared.config import CFG, Settings, load_settings


def test_repo_config_loads_edge_and_server():
    assert CFG.edge.max_nodes >= 1
    assert CFG.edge.window_s > 0
    assert CFG.edge.escalate_min_confidence == 0.50
    assert CFG.alert.cooldown_s == 3.0
    assert CFG.server.senior_wait_s == 60.0
    assert CFG.server.family_wait_s == 60.0
    assert CFG.server.careline_at_s == 180.0


def test_missing_file_uses_defaults(tmp_path):
    settings = load_settings(tmp_path / "nope.yaml")
    assert isinstance(settings, Settings)
    assert settings.edge.udp_port == 9000
    assert settings.server.http_port == 8001

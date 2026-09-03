import csv
from pathlib import Path

from scripts.record import write_take


def test_write_take_csv_and_metadata(tmp_path, monkeypatch):
    import scripts.record as rec

    monkeypatch.setattr(rec, "DATA", tmp_path)
    rows = [
        {"t": 1, "ax": 0.1, "ay": 0.0, "az": 0.2, "mag": 0.22, "db": -40.0},
        {"t": 21, "ax": 0.0, "ay": 0.0, "az": 0.0, "mag": 0.01, "db": -41.0},
    ]
    path = write_take("heeldrop", rows, {"floor": "tile", "room": "kitchen", "distance_m": 1.5})
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "t,ax,ay,az,mag,db"
    with path.open(encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 2
    assert got[0]["mag"] == "0.22"
    meta = Path(tmp_path / "metadata.csv")
    body = meta.read_text(encoding="utf-8")
    assert "heeldrop_01.csv" in body
    assert "tile" in body

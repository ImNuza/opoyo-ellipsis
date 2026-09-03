#!/usr/bin/env python3
"""Run this before the demo. Every check that can silently degrade a live run.

    python -m scripts.preflight
"""
from __future__ import annotations
import os, socket, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OK, BAD = "  ok  ", " FAIL "
fails = 0


def check(name, cond, hint=""):
    global fails
    print(f"[{OK if cond else BAD}] {name}")
    if not cond:
        fails += 1
        if hint:
            print(f"         {hint}")
    return cond


print("OPOYO preflight\n")

check("python >= 3.10", sys.version_info >= (3, 10))
try:
    import numpy, scipy, sklearn, joblib, fastapi, pydantic          # noqa: F401
    check("core deps import", True)
except Exception as e:
    check("core deps import", False, f"{e}  ->  pip install -r requirements.txt")

tf_ok = True
try:
    import tensorflow_hub                                            # noqa: F401
except Exception as e:
    tf_ok = False
check("tensorflow_hub import", tf_ok,
      "without it the edge silently uses the vibration-only head (AP 0.397, not 0.853)")

cache = ROOT / "models" / "tfhub"
check("YAMNet cached offline", cache.is_dir() and any(cache.iterdir()),
      "first inference would download from tfhub.dev on venue wifi")

heads = ["mag_head.joblib", "yamnet_head.joblib", "fuse_head.joblib"]
have = [h for h in heads if (ROOT / "models" / h).exists()]
check(f"model heads present ({len(have)}/3)", len(have) == 3,
      "run: python -m train.fit")

takes = list((ROOT / "data" / "takes").glob("*/*.csv"))
check(f"take data present ({len(takes)} csv)", len(takes) > 50)

env = ROOT / ".env"
check(".env exists", env.exists(), "cp .env.example .env and fill the Telegram ids")
if env.exists():
    txt = env.read_text()
    check("TELEGRAM_BOT_TOKEN set", "TELEGRAM_BOT_TOKEN=" in txt
          and txt.split("TELEGRAM_BOT_TOKEN=")[1].split("\n")[0].strip() != "",
          "escalation will no-op without it")
    check("EDGE_ENABLE_UDP=1", "EDGE_ENABLE_UDP=1" in txt,
          "phones will stream into the void")

for port in (9000, 9001):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", port)); free = True
    except OSError:
        free = False
    finally:
        s.close()
    check(f"UDP :{port} free", free, "something else is already bound")

if tf_ok and len(have) == 3:
    try:
        from edge.infer import load_runtime
        clf = load_runtime()
        check("live classifier is FusionCnn", type(clf).__name__ == "FusionCnn",
              f"got {type(clf).__name__} -- the demo would run on the weak head")
    except Exception as e:
        check("live classifier loads", False, str(e)[:110])

print(f"\n{'ALL CLEAR' if not fails else str(fails) + ' CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)

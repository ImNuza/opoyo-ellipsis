from opoyo.config import CFG
from opoyo.pipeline import LivePipeline


def test_ingest_waits_1_5s_then_cuts_2s_clip():
    clock = [0.0]
    seen = []
    pipe = LivePipeline(CFG, now_fn=lambda: clock[0])
    orig = pipe.machine.on_trigger

    def spy(window, audio, room="", node_id="", quiet_rms=0.0):
        seen.append(
            {
                "t": clock[0],
                "n": int(window.size),
                "mean": float(window.mean()),
                "pending": pipe._pending_t,
            }
        )
        return orig(window, audio, room=room, node_id=node_id, quiet_rms=quiet_rms)

    pipe.machine.on_trigger = spy  # type: ignore[method-assign]
    dt = 1.0 / pipe.fs
    fire_t = None
    # mag_g=1.0 is a gravity-like DC (9.81 m/s²). HP must remove it before extract.
    g_dc = 1.0
    for i in range(int(pipe.fs * 8)):
        mag = g_dc + (2.0 if int(pipe.fs * 3) <= i < int(pipe.fs * 3.25) else 0.0)
        if pipe._pending_t is not None and fire_t is None:
            fire_t = pipe._pending_t
        pipe.feed(0.0, 0.0, mag, db=-40.0, mag_g=mag)
        clock[0] += dt
        if seen:
            break
    assert fire_t is not None, "trigger never armed"
    assert seen, "on_trigger never ran"
    assert seen[0]["t"] - fire_t >= float(CFG.window.post_s) - dt
    assert seen[0]["n"] == pipe.pre_n + pipe.post_n
    assert abs((pipe.pre_n + pipe.post_n) / pipe.fs - 2.0) < 1e-9
    assert abs(seen[0]["mean"]) < 1.0, "clip still has DC; high-pass did not feed the ring"

import sys, types
from ontic_lib import tracking

class FakeTrackio(types.ModuleType):
    def __init__(self):
        super().__init__("trackio")
        self.calls = []
        self.init = lambda **kw: self.calls.append(("init", kw)) or self
        self.log = lambda metrics, step=None: self.calls.append(("log", metrics, step))
        self.finish = lambda: self.calls.append(("finish",))

def test_trackio_dir_defaults_into_output(monkeypatch, tmp_path):
    fake = FakeTrackio()
    monkeypatch.setitem(sys.modules, "trackio", fake)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRACKIO_DIR", raising=False)
    t = tracking.init("saliency", {"lr": 0.1})
    import os
    assert os.environ["TRACKIO_DIR"].endswith("output")
    t.log({"loss": 1.0}, step=1)
    t.finish()
    assert ("log", {"loss": 1.0}, 1) in fake.calls

def test_wandb_optional_and_never_fatal(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "trackio", FakeTrackio())
    bad = types.ModuleType("wandb")
    bad.init = lambda **kw: (_ for _ in ()).throw(RuntimeError("down"))
    monkeypatch.setitem(sys.modules, "wandb", bad)
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("ONTIC_WANDB_RUN_ID", "abc")
    monkeypatch.chdir(tmp_path)
    t = tracking.init("p")                    # must not raise
    t.log({"x": 1})
    t.finish()

import src.service as service


class TestModelCache:
    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def test_same_model_loaded_once(self, monkeypatch):
        loads = []

        def fake_load(name, device=None):
            loads.append(name)
            return f"model:{name}"

        monkeypatch.setattr(service.whisper, "load_model", fake_load)
        a = service.get_whisper_model("base", "cpu")
        b = service.get_whisper_model("base", "cpu")
        assert a is b
        assert loads == ["base"]  # loaded exactly once

    def test_model_change_evicts_and_reloads(self, monkeypatch):
        loads = []
        collected = []
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: (loads.append(name) or f"model:{name}"))
        monkeypatch.setattr(service.gc, "collect", lambda: collected.append(True))

        service.get_whisper_model("base", "cpu")
        service.get_whisper_model("large-v3", "cpu")
        assert loads == ["base", "large-v3"]
        assert collected  # eviction ran gc.collect at least once

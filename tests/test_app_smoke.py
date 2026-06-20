def test_app_builds_blocks():
    import app

    demo = app.build_ui()
    # gradio Blocks exposes .launch; assert we constructed something launchable
    assert hasattr(demo, "launch")


def test_history_row_mapping_handles_empty():
    import app

    assert app._history_rows([]) == []


def test_monotonic_progress_drops_backward_updates():
    import app

    seen = []
    cb = app._make_monotonic_progress(lambda f, desc=None: seen.append(f))
    for f in (0.0, 0.5, 0.3, 1.0, 0.9):  # 0.3 and 0.9 are backward -> dropped
        cb(f)
    assert seen == [0.0, 0.5, 1.0]

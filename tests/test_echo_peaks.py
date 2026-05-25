from pathlib import Path

import numpy as np
import pandas as pd

from echopress.core.echo_peaks import EchoPeakConfig, run_echo_peak_detection


def _write_first_peak_index(path: Path, first_peaks: list[int]) -> None:
    pd.DataFrame(
        {
            "path": [str(path)] * len(first_peaks),
            "first_peak_idx": first_peaks,
            "pressure_value": [1.0] * len(first_peaks),
            "file_index": [0] * len(first_peaks),
            "macro_window_index": list(range(len(first_peaks))),
        }
    ).to_csv(path.parent / "first_peak_index.csv", index=False)


def test_echo_window_end_uses_next_first_peak(monkeypatch, tmp_path: Path):
    detection_dir = tmp_path / "det"
    output_dir = tmp_path / "out"
    detection_dir.mkdir()

    first_peaks = [10, 20, 40]
    _write_first_peak_index(detection_dir / "sig.csv", first_peaks)

    sig = np.zeros(80, dtype=float)
    sig[[14, 26, 46]] = [5.0, 6.0, 7.0]

    class DummyOstream:
        channels = sig
        timestamps = np.arange(80, dtype=float) * 1e-6

    monkeypatch.setattr("echopress.core.echo_peaks.load_ostream", lambda *_args, **_kwargs: DummyOstream())
    monkeypatch.setattr("echopress.core.echo_peaks.write_resolved_config", lambda *_args, **_kwargs: None)

    cfg = EchoPeakConfig(
        detection_dir=detection_dir,
        output_dir=output_dir,
        use_registered=False,
        fallback_to_t_global_window_end=True,
    )
    run_echo_peak_detection(cfg)

    windows = pd.read_csv(output_dir / "echo_window_index.csv")
    assert windows["window_end_idx_exclusive"].tolist() == [20, 40, 55]
    assert windows["window_end_source"].tolist() == ["next_first_peak", "next_first_peak", "t_global_fallback"]


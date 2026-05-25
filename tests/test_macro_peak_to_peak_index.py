import pandas as pd

from echopress.core.macro_detector import build_peak_to_peak_window_index


def test_build_peak_to_peak_window_index_yields_four_windows_per_file_for_five_peaks():
    rows = []
    for file_index, file_name in enumerate(("f1.npz", "f2.npz")):
        for peak_idx in (100, 200, 300, 400, 500):
            rows.append(
                {
                    "path": f"/tmp/{file_name}",
                    "file": file_name,
                    "pressure_value": 10.0 + file_index,
                    "file_index": file_index,
                    "first_peak_idx": peak_idx,
                }
            )

    first_peak_df = pd.DataFrame(rows)

    p2p_df = build_peak_to_peak_window_index(first_peak_df, t_global_samples=100.0)

    per_file = p2p_df.groupby("path").size().to_dict()
    assert per_file == {"/tmp/f1.npz": 4, "/tmp/f2.npz": 4}
    assert (p2p_df["window_len_samples"] == 100).all()

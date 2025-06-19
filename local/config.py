from types import SimpleNamespace
from pathlib import Path


def default_cfg() -> SimpleNamespace:
    """Возвращает конфигурацию по умолчанию для ForwardBendMonitor."""
    return SimpleNamespace(
        backend="onnxruntime",
        device="cuda",
        kpt_threshold=0.6,
        straight_knee_threshold=140,
        hold_duration=2.0,
        finger_px_threshold=60.0,
        results_dir=str(Path(__file__).with_name("results")),
        save_format="mp4",
        log_file="monitor.log",
        output_duration=60,
    ) 
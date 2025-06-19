from types import SimpleNamespace
from pathlib import Path
from datetime import datetime


def default_cfg() -> SimpleNamespace:
    """Формирует конфигурацию по умолчанию для `ForwardBendMonitor`.

    Лог-файл кладётся в каталог `logs/` с названием `monitor-YYYYMMDD.log`.
    """
    today = datetime.now().strftime("%Y%m%d")
    # Каталоги для результатов и логов располагаются рядом с данным файлом
    root_dir = Path(__file__).resolve().parent
    results_dir = root_dir / "results"
    logs_dir = root_dir / "logs"
    log_file = logs_dir / f"monitor-{today}.log"

    return SimpleNamespace(
        backend="onnxruntime",
        device="cuda",
        mode="balanced",  # режим Wholebody («balanced», «lightweight», …)
        kpt_threshold=0.6,
        straight_knee_threshold=140,
        hold_duration=2.0,
        finger_px_threshold=60.0,
        wrist_tolerance=20,  # допустимое вертикальное смещение запястья, px
        results_dir=str(results_dir),
        save_format="mp4",
        log_file=str(log_file),
        output_duration=60,
        post_capture_seconds=2,  # дополнительно захватывать после завершения, сек
    ) 
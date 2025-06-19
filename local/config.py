from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MonitorConfig:
    """Параметры работы `ForwardBendMonitor`. Используется вместо `SimpleNamespace`."""

    # — Runtime options —
    backend: str = "onnxruntime"
    device: str = "cuda"
    mode: str = "balanced"  # режим Wholebody («balanced», «lightweight», …)

    # — Пороговые значения —
    kpt_threshold: float = 0.6
    straight_knee_threshold: int = 140
    hold_duration: float = 2.0
    finger_px_threshold: float = 60.0
    wrist_tolerance: int = 20  # допустимое вертикальное смещение запястья, px

    # — Вывод и логирование —
    results_dir: str = ""
    save_format: str = "mp4"
    log_file: str = ""
    output_duration: int = 60
    post_capture_seconds: int = 2  # дополнительно захватывать после завершения, сек

    # — Прочее —
    headless: bool = False  # если True — не показывать окна cv2.imshow


def default_cfg() -> MonitorConfig:
    """Формирует конфигурацию по умолчанию для `ForwardBendMonitor`.

    Лог-файл кладётся в каталог `logs/` с названием `monitor-YYYYMMDD.log`.
    """

    today = datetime.now().strftime("%Y%m%d")
    # Каталоги для результатов и логов располагаются рядом с данным файлом
    root_dir = Path(__file__).resolve().parent
    results_dir = root_dir / "results"
    logs_dir = root_dir / "logs"
    log_file = logs_dir / f"monitor-{today}.log"

    return MonitorConfig(
        results_dir=str(results_dir),
        log_file=str(log_file),
    ) 
from config import default_cfg
from monitor import ForwardBendMonitor


def main() -> None:
    cfg = default_cfg()
    monitor = ForwardBendMonitor(cfg)
    monitor.run()


if __name__ == "__main__":
    main()

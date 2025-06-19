from pathlib import Path
import logging

import pygame

# Инициализируем аудио при первом импорте
pygame.mixer.init()

SOUND_START = "audio/start.mp3"
SOUND_SUCCESS = "audio/success.mp3"
SOUND_RESET = "audio/reset.mp3"

__all__ = [
    "SOUND_START",
    "SOUND_SUCCESS",
    "SOUND_RESET",
    "play_sound",
]


def play_sound(path: str) -> None:
    """Безопасное воспроизведение звука по указанному пути относительно этого файла."""
    # Если путь относительный — считаем его относительным к каталогу текущего файла
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    if not p.exists():
        logging.warning("Sound file not found: %s", p)
        return
    try:
        pygame.mixer.music.load(str(p))
        pygame.mixer.music.play()
    except Exception as exc:
        logging.warning("Pygame audio error: %s", exc) 
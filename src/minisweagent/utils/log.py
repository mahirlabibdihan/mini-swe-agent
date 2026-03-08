import logging
from pathlib import Path

from rich.logging import RichHandler


def _setup_root_logger() -> None:
    logger = logging.getLogger("minisweagent")
    logger.setLevel(logging.DEBUG)
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

def _setup_instance_logger() -> None:
    logger = logging.getLogger(f"minisweagent_instance")
    logger.setLevel(logging.DEBUG)
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    _formatter = logging.Formatter("%(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    

def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    logger = logging.getLogger("minisweagent")

    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if print_path:
        print(f"Logging to '{path}'")

def set_instance_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    logger = logging.getLogger(f"minisweagent_instance")

     # Remove existing FileHandlers
    for h in logger.handlers[:]:  # copy list to avoid modifying while iterating
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if print_path:
        print(f"Logging to '{path}'")


_setup_root_logger()
_setup_instance_logger()
logger = logging.getLogger("minisweagent")
instance_logger = logging.getLogger("minisweagent_instance")

__all__ = ["logger", "instance_logger"]

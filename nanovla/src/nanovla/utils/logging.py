# Copyright 2024 The nanoVLA Team. All rights reserved.
# Adapted from HuggingFace Transformers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Logging utilities for nanoVLA.
"""

import logging as _logging
import os
import sys
import threading
from logging import CRITICAL, DEBUG, ERROR, FATAL, INFO, NOTSET, WARN, WARNING  # noqa
from typing import Optional


_lock = threading.Lock()
_default_handler: Optional[_logging.Handler] = None

log_levels = {
    "debug": DEBUG,
    "info": INFO, 
    "warning": WARNING,
    "error": ERROR,
    "critical": CRITICAL,
}

_default_log_level = WARNING


def _get_default_logging_level():
    """Get the default logging level."""
    env_level_str = os.getenv("NANOVLA_VERBOSITY", None)
    if env_level_str:
        if env_level_str.lower() in log_levels:
            return log_levels[env_level_str.lower()]
        else:
            _logging.getLogger().warning(
                f"Unknown option NANOVLA_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> _logging.Logger:
    return _logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = _logging.StreamHandler()  # Set sys.stderr as stream.
        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(NOTSET)
        _default_handler = None


def get_log_levels_dict():
    return log_levels


def captureWarnings(capture):
    """
    Calls the `captureWarnings` method from the logging library to enable management of the warnings emitted by the
    `warnings` library.

    Read more about this method here:
    https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module

    All warnings will be logged through the `py.warnings` logger.

    Careful: this method also adds a handler to the `py.warnings` logger if it doesn't already have one, and updates the
    logging level of that logger to the library's root logger.
    """
    logger = _logging.getLogger("py.warnings")

    if not logger.handlers:
        logger.addHandler(_default_handler)

    logger.setLevel(_get_library_root_logger().level)

    _logging.captureWarnings(capture)


def get_logger(name: Optional[str] = None) -> _logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom nanovla module.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return _logging.getLogger(name)


def get_verbosity() -> int:
    """
    Return the current level for the 🤗 nanoVLA' root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    🤗 nanoVLA has following logging levels:

    - 50: `nanovla.logging.CRITICAL` or `nanovla.logging.FATAL`
    - 40: `nanovla.logging.ERROR`
    - 30: `nanovla.logging.WARNING` or `nanovla.logging.WARN`
    - 20: `nanovla.logging.INFO`
    - 10: `nanovla.logging.DEBUG`

    </Tip>
    """

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the 🤗 nanoVLA' root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `nanovla.logging.CRITICAL` or `nanovla.logging.FATAL`
            - `nanovla.logging.ERROR`
            - `nanovla.logging.WARNING` or `nanovla.logging.WARN`
            - `nanovla.logging.INFO`
            - `nanovla.logging.DEBUG`
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the verbosity to the `WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Set the verbosity to the `DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """Set the verbosity to the `ERROR` level."""
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    """Disable the default handler of the 🤗 nanoVLA' root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the 🤗 nanoVLA' root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def add_handler(handler: _logging.Handler) -> None:
    """adds a handler to the HuggingFace Transformers' root logger."""

    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: _logging.Handler) -> None:
    """removes given handler from the HuggingFace Transformers' root logger."""

    _configure_library_root_logger()

    assert handler is not None and handler not in _get_library_root_logger().handlers
    _get_library_root_logger().removeHandler(handler)


def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the HuggingFace Transformers' default handler to
    prevent double logging if the root logger has been configured.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every 🤗 nanoVLA' logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NO] TIME >> MESSAGE
    ```

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = _logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)


def reset_format() -> None:
    """
    Resets the formatting for 🤗 nanoVLA' loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)


def warning_advice(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but if env var NANOVLA_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv("NANOVLA_NO_ADVISORY_WARNINGS", False)
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)


_logging.Logger.warning_advice = warning_advice


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        if _tqdm_available:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None

    def get_lock(self):
        return self._lock


try:
    import tqdm as tqdm_lib

    _tqdm_available = True
except ImportError:
    _tqdm_available = False

tqdm = _tqdm_cls()


# Configure the library root logger.
_configure_library_root_logger()

# Create the main logger that will be imported
default_logger = get_logger()





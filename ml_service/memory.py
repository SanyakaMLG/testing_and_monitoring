import gc
import logging

try:
    import ctypes
except ImportError:
    ctypes = None


def release_process_memory(logger: logging.Logger | None = None) -> None:
    gc.collect()

    if ctypes is None:
        return

    try:
        libc = ctypes.CDLL('libc.so.6')
        malloc_trim = getattr(libc, 'malloc_trim', None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        if logger is not None:
            logger.debug('Unable to trim process memory', exc_info=True)

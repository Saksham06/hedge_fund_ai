"""
Two-tier cache:
  1. In-memory LRU for hot paths (sub-second access)
  2. Disk cache via diskcache for cross-process / cross-restart persistence

Falls back to in-memory-only if diskcache is unavailable.
"""
import functools
import logging
import os
import time

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "state", ".cache")

try:
    import diskcache
    _disk = diskcache.Cache(_CACHE_DIR)
    _DISK_AVAILABLE = True
except Exception:
    _disk = None
    _DISK_AVAILABLE = False

# In-memory tier: {key: (value, expiry_ts)}
_mem: dict = {}


def _mem_get(key):
    entry = _mem.get(key)
    if entry and time.time() < entry[1]:
        return entry[0], True
    return None, False


def _mem_set(key, value, ttl):
    _mem[key] = (value, time.time() + ttl)


def simple_cache(ttl: int = 300, disk: bool = True):
    """
    Decorator that caches function results.
    ttl: seconds until entry expires
    disk: also persist to diskcache (survives restarts)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__module__}.{func.__qualname__}:{args!r}:{sorted(kwargs.items())!r}"

            # L1: memory
            val, hit = _mem_get(key)
            if hit:
                return val

            # L2: disk
            if disk and _DISK_AVAILABLE:
                try:
                    val = _disk.get(key, default=None)
                    if val is not None:
                        _mem_set(key, val, ttl)
                        return val
                except Exception:
                    pass

            # Miss: compute
            result = func(*args, **kwargs)

            _mem_set(key, result, ttl)
            if disk and _DISK_AVAILABLE:
                try:
                    _disk.set(key, result, expire=ttl)
                except Exception:
                    pass

            return result
        return wrapper
    return decorator


def clear_all():
    """Wipe both cache tiers."""
    _mem.clear()
    if _DISK_AVAILABLE:
        try:
            _disk.clear()
        except Exception:
            pass

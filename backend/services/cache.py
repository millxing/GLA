import threading
from cachetools import TTLCache
from config import CACHE_TTL_SECONDS, CACHE_MAX_SIZE

cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
_lock = threading.Lock()

def get_cache_key(func_name: str, *args, **kwargs) -> str:
    key_parts = [func_name]
    for arg in args:
        key_parts.append(str(arg))
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    return ":".join(key_parts)

def get_cached(key: str):
    with _lock:
        return cache.get(key)

def set_cached(key: str, value):
    with _lock:
        cache[key] = value

def clear_cache():
    with _lock:
        cache.clear()

"""Redis async cache wrapper."""
import json
import logging
from redis.asyncio import from_url
from rainfall_service.config import settings

logger = logging.getLogger(__name__)
_redis = None


async def init_redis():
    global _redis
    _redis = await from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    logger.info("Redis connected.")


def get_redis():
    if _redis is None:
        raise RuntimeError("Redis not initialized.")
    return _redis


async def cache_set(key: str, value: dict, ttl: int = 300):
    try:
        r = get_redis()
        await r.setex(key, ttl, json.dumps(value))
    except Exception as e:
        logger.warning("Cache set failed: %s", e)


async def cache_get(key: str):
    try:
        r = get_redis()
        data = await r.get(key)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return None
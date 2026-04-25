import os
import redis
from rq import Queue

# 🔧 Environment-based config (important for EC2 + Docker)
REDIS_HOST = os.getenv("REDIS_HOST", "redis")   # "redis" for Docker, "localhost" for local
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# 🔗 Create Redis connection
redis_conn = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=False  # important for binary data
)

# 🎯 Create queue
queue = Queue(
    "ocr_queue",            # queue name
    connection=redis_conn,
    default_timeout=300     # 5 minutes timeout (important for ML jobs)
)
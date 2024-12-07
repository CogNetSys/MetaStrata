from redis.asyncio import Redis
from supabase import create_client
import os
from config import REDIS_PASSWORD, REDIS_ENDPOINT, SUPABASE_URL, SUPABASE_KEY
from utils import add_log, LOG_QUEUE, logger

# Initialize Redis client
redis = Redis(
    host=REDIS_ENDPOINT,
    port=6379,
    password=REDIS_PASSWORD,
    decode_responses=True,  # Ensures responses are decoded as strings
    ssl=True,  # Enable SSL/TLS for secure Redis connection
)

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# /app/database.py

import os  # Add this import
from redis.asyncio import Redis
from supabase import create_client, Client
from app.config import settings
from pydantic import SecretStr  # Ensure SecretStr is imported

# Redis Initialization
redis = Redis(
    host=settings.REDIS.REDIS_ENDPOINT,
    port=6379,
    password=settings.REDIS.REDIS_PASSWORD.get_secret_value(),  # Extract the plain string
    decode_responses=True,
    ssl=True  # Enable SSL/TLS
)

# Supabase Client Initialization
supabase_url = settings.DATABASE.SUPABASE_URL
supabase_key = settings.DATABASE.SUPABASE_KEY

# Check if Supabase URL and Key are available
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in the environment variables")

# If supabase_key is a SecretStr, access its string value
if isinstance(supabase_key, SecretStr):
    supabase_key = supabase_key.get_secret_value()

# Clean URL and key just in case there are any extra spaces
cleaned_supabase_url = supabase_url.strip()
cleaned_supabase_key = supabase_key.strip()

# Initialize Supabase Client with cleaned values
supabase: Client = create_client(cleaned_supabase_url, cleaned_supabase_key)

# Secrets Management System with Pydantic and Doppler

This document explains how your application manages secrets securely using **Pydantic** for validation and **Doppler** for secrets management. The system is designed for both local development and production, ensuring security, validation, and ease of use.

---

## Overview

1. **Doppler:**  
   Doppler serves as the central hub for managing secrets, injecting environment variables into your application at runtime.
   
2. **Pydantic:**  
   Pydantic is used to validate and load these environment variables into structured settings classes, ensuring that your application configuration is secure and type-safe.

---

## How It Works

1. **Doppler Injects Secrets:**  
   Secrets are securely stored in Doppler and injected as environment variables when the application runs.

2. **Pydantic Settings Classes:**  
   Each module (e.g., Database, Redis) has its own `BaseSettings` class, which:
   - Maps environment variables to fields.
   - Validates required fields.
   - Provides default values for optional fields.

3. **Centralized Access:**  
   All settings are encapsulated in a global `settings` object, allowing easy access throughout the application.

---

## Secrets Structure

### **Example Secrets**

| Name                | Description                          | Required |
|---------------------|--------------------------------------|----------|
| `SUPABASE_KEY`      | Supabase API Key                    | Yes      |
| `SUPABASE_URL`      | Supabase URL                        | Yes      |
| `GROQ_API_KEY`      | GROQ API Key                        | Yes      |
| `REDIS_ENDPOINT`    | Redis Host Endpoint                 | Yes      |
| `REDIS_PASSWORD`    | Redis Password                      | Yes      |
| `AUTH_TOKEN`        | Authentication Token                | Yes      |
| `E2B_API_KEY`       | API Key for E2B                     | Yes      |
| `LOGFIRE_API_KEY`   | Logfire API Key                     | Yes      |
| `LOGFIRE_ENDPOINT`  | Logfire Endpoint                    | No       |
| `LOGFIRE_ENABLED`   | Enable Logfire (True/False)         | No       |

---

## Secrets Management with Doppler

### **Setting Up Doppler**

1. **Install Doppler CLI**  
   Follow instructions at [Doppler CLI Installation](https://docs.doppler.com/docs/cli).

2. **Authenticate CLI**  
   ```bash
   doppler login
   ```

3. **Set Up Project**  
   Create a Doppler project and configure environments:
   ```bash
   doppler setup
   ```

4. **Add Secrets to Doppler**  
   Use the CLI or dashboard to add secrets:
   ```bash
   doppler secrets set SUPABASE_KEY="your-supabase-key"
   doppler secrets set REDIS_ENDPOINT="your-redis-endpoint"
   ```

5. **Run Application with Doppler**  
   Inject secrets into your application at runtime:
   ```bash
   doppler run -- python3 run.py
   ```

---

## Configuration with Pydantic

### **How Pydantic Loads Secrets**

- **Environment Variable Mapping:**  
  Fields in each `BaseSettings` class map to specific environment variables using the `Field` argument `env`.

- **Validation:**  
  - Required fields (`Field(..., env="VAR")`) raise errors if missing.
  - Default values are used for optional fields (`Field(default_value, env="VAR")`).

### **Example Settings Class**

```python
from pydantic import BaseSettings, SecretStr, Field

class RedisSettings(BaseSettings):
    REDIS_ENDPOINT: str = Field(..., env="REDIS_ENDPOINT")  # Required
    REDIS_PASSWORD: SecretStr = Field(..., env="REDIS_PASSWORD")  # Required

    class Config:
        env_file = ".env"  # Optional: Support for local .env files
```

---

## Local Development

For local development without Doppler, use a `.env` file to store secrets. Example:

### **.env File**
```plaintext
SUPABASE_KEY=your-local-supabase-key
SUPABASE_URL=https://local.supabase.url
REDIS_ENDPOINT=redis://localhost:6379
REDIS_PASSWORD=local-redis-password
```

The settings classes automatically load this file during local runs.

---

## Example: Accessing Secrets

### **Global Settings Instance**

All settings are encapsulated in a centralized `settings` object:
```python
from settings import settings

# Access Redis settings
redis_host = settings.REDIS.REDIS_ENDPOINT
redis_password = settings.REDIS.REDIS_PASSWORD.get_secret_value()
```

### **Validation Errors**

If a required secret is missing, Pydantic raises a clear error:
```
pydantic.error_wrappers.ValidationError: 1 validation error for RedisSettings
REDIS_ENDPOINT
  field required (type=value_error.missing)
```

---

## Advantages of This Setup

1. **Security:**  
   Secrets are never hardcoded; Doppler manages their storage and injection.

2. **Validation:**  
   Pydantic ensures all required secrets are provided and correctly formatted.

3. **Portability:**  
   Supports both Doppler for production and `.env` for local development.

4. **Ease of Use:**  
   The `settings` object provides a simple interface to access all configuration values.

---

## Debugging Tips

1. **Verify Secrets in Doppler:**  
   Use the CLI to check secrets:
   ```bash
   doppler secrets
   ```

2. **Debug Environment Variables:**  
   Print injected variables:
   ```bash
   doppler run -- env
   ```

3. **Add Debug Logs in Code:**  
   Log settings to ensure they load correctly:
   ```python
   print("Redis Endpoint:", settings.REDIS.REDIS_ENDPOINT)
   print("Supabase URL:", settings.DATABASE.SUPABASE_URL)
   ```

---

## Summary

This system combines **Doppler** for secure secrets management with **Pydantic** for validation, ensuring your application configuration is secure, validated, and easy to maintain. It is designed for seamless operation across development, staging, and production environments.
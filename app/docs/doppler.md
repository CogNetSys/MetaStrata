# Doppler Setup and Usage Guide

This document provides a comprehensive guide for setting up, configuring, and using Doppler to manage secrets in your application.

---

## **What is Doppler?**

Doppler is a secrets management tool that centralizes sensitive configuration data such as API keys, database credentials, and other environment variables. It provides a secure, scalable, and developer-friendly way to manage secrets across environments.

---

## **Setting Up Doppler**

### **1. Install Doppler CLI**

To use Doppler, you need to install the Doppler CLI:

#### macOS (via Homebrew)
brew install dopplerhq/cli/doppler

#### Linux (via Script)
curl -sLf --retry 3 --retry-delay 2 https://cli.doppler.com/install.sh | sh

#### Windows
Download the Doppler CLI from the [Doppler Downloads page](https://www.doppler.com/docs/cli/install).

---

### **2. Log In to Doppler**

Authenticate the Doppler CLI with your account:
doppler login

This will open a browser window to log in. After logging in, the CLI will be authenticated.

---

### **3. Create a Doppler Project**

1. Navigate to the [Doppler Dashboard](https://dashboard.doppler.com).
2. Create a new project for your application (e.g., `cognetics-architect`).
3. Add environments like `development`, `staging`, and `production`.

---

### **4. Add Secrets to Doppler**

For each environment, add the required secrets. For example:
- SUPABASE_KEY
- SUPABASE_URL
- GROQ_API_KEY
- REDIS_ENDPOINT
- REDIS_PASSWORD
- AUTH_TOKEN

You can add secrets via the Doppler Dashboard or the CLI:
doppler secrets set SUPABASE_KEY="your-supabase-key"
doppler secrets set REDIS_PASSWORD="your-redis-password"

---

### **5. Link Doppler to Your Project**

To link your local project to Doppler:
doppler setup

Follow the prompts to select your project and environment. This creates a `.doppler.yaml` file in your project directory.

---

## **Using Doppler**

### **1. Inject Secrets into Your Application**

Run your application with Doppler managing the environment variables:
doppler run -- python3 run.py
doppler run --debug -- python3 run.py

This injects secrets as environment variables into your application at runtime.

---

### **2. Debugging Doppler Secrets**

To see all secrets available in the current environment:
doppler secrets

To debug environment variable injection:
doppler run -- env

---

### **3. Sync Doppler with Deployment Platforms**

Doppler supports syncing secrets directly to platforms like Vercel, AWS, and Docker. For example:

#### Syncing with Vercel
1. Go to the **Integrations** tab in the Doppler Dashboard.
2. Select **Vercel** and follow the setup instructions.
3. Doppler will sync secrets to Vercel's environment variables automatically.

---

## **Best Practices**

1. **Avoid .env Files:**
   Doppler eliminates the need for `.env` files, reducing security risks.

2. **Rotate Secrets Regularly:**
   Use Doppler to rotate secrets without downtime or redeployments.

3. **Use Multiple Environments:**
   Configure separate environments (e.g., `development`, `staging`, `production`) to isolate secrets.

4. **Grant Access by Role:**
   Use Doppler's role-based access control (RBAC) to manage team permissions.

---

## **Troubleshooting**

### Missing Environment Variables
- Ensure you are running the application with `doppler run`.
- Verify secrets exist in Doppler with:
  doppler secrets

### Debugging Doppler CLI
Run the CLI in debug mode:
doppler run --debug -- python3 run.py

---

## **Resources**
- [Doppler Documentation](https://www.doppler.com/docs)
- [Doppler CLI Commands](https://www.doppler.com/docs/cli)

With Doppler, you can securely manage and inject secrets into your application, ensuring your sensitive data is safe and easy to maintain.
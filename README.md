### README.md

# 🧠 Takata et al. Experiment Simulation

This project implements an enhanced version of the **Takata et al. experiment** as described in their [research paper](https://arxiv.org/pdf/2411.03252). It simulates autonomous agents moving in a grid, exchanging messages, and building memories based on their surroundings.

The application is built using **Python** with **FastAPI**, uses **Supabase** as the database, and **Redis** for caching. It also includes a WebSocket client for interactive logging.

Explore the spontaneous emergence of individuality, social norms, and collective intelligence in multi-agent systems powered by Large Language Models (LLMs). This project replicates and extends the groundbreaking work from the paper "Spontaneous Emergence of Agent Individuality Through Social Interactions in LLM-Based Communities" by Ryosuke Takata et al.

With agents starting from a uniform state, this simulation demonstrates how personalities, emotions, and behaviors evolve organically through cooperative communication. Using Supabase and Redis, the application implements a grid-based environment where agents interact via natural language, generating insights into emergent phenomena such as hashtags, hallucinations, and shared narratives.

---

## 🌟 Features

- **No predefined individuality:** Agents begin as homogeneous entities, with unique traits emerging through interaction.
- **Emergent phenomena:** Witness the rise of shared norms, emotional synchronicity, and creative hallucinations in a simulated community.
- **Scalable architecture:** Built on Python's FastAPI, with Supabase for persistence and Redis for caching, ensuring efficient simulation management.
- **Research-focused:** A platform for studying the dynamics of collective intelligence and individuality in LLM-based agent communities.

---

## 📂 File Structure

- **`main.py`**: Core FastAPI application with endpoints and simulation logic.
- **`/static/js/websocket_client.js`**: WebSocket client for logging.
- **`requirements.txt`**: Python dependencies.

---

## 🚀 Installation

Follow these steps to set up the application locally:

### 1️⃣ Prerequisites

- Python 3.9+
- free Vercel account
- free Upstash Redis account
- free Supabase account
- free Groq Cloud account
- Node.js (optional, for WebSocket client)

---

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo-url/takata-simulation.git
cd takata-simulation
```

---

### 3️⃣ Set Up a Virtual Environment

```bash
python -m venv takata-simulation
source takata-simulation/bin/activate  # For Linux/Mac
takata-simulation\Scripts\activate     # For Windows
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5️⃣ Configure Environment Variables

Create a `.env` file in the project directory and add your configuration:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key
GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
REDIS_ENDPOINT=your_redis_endpoint_without_https (dont insert the https:// part, ie "cute-crawdaddy-23143.upstash.io")
REDIS_PASSWORD=your_redis_password
```

---

### 6️⃣ Run Redis Locally

Start a Redis instance (ensure it's reachable via the `REDIS_ENDPOINT` in the code).

```bash
redis-server
```

---

### 7️⃣ Run the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`.

---

### 8️⃣ Access Endpoints

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Vercel Swagger UI**: http://<your-app>.vercel.app/docs

---

## 💡 Usage

- Use the **Redis `MONITOR`** feature to view live logs of agent **messages**, **memories**, and **movements**.
- Reset or initialize the simulation using `/reset` and `/start` endpoints.
- Execute steps with `/step` to observe agent behavior.

---

## 🛠 Deployment to Vercel

1. Install the [Vercel CLI](https://vercel.com/docs/cli).
2. Deploy the application:

   ```bash
   vercel deploy
   ```
3. After you deploy, click the link to your deployment.
4. Append "/docs" to the URL and click "enter."
5. Use the FastAPI/Swagger UI to control the experiment.
6. You can view the websocket stream at the bottom of the page.
    - If the websocket stream ends, refresh the page to restart it.
---

## 🔑 Key Commands

- **Start Simulation**: Use the `/start` endpoint.
- **Reset Simulation**: Use the `/reset` endpoint.
- **Perform Steps**: Use `/step` with a JSON payload specifying the number of steps:

  ```json
  {
    "steps": 10
  }
  ```

---

## ⚙️ Redis Monitoring

Run the following command to view real-time agent logs:

```bash
redis-cli monitor
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

### 🎉 Enjoy exploring agent behaviors and distributed simulations!
# AI Textile Machine Health Intelligence System

A modern **web dashboard** built with Streamlit for predictive maintenance in textile manufacturing.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Then open the browser URL shown in terminal (typically `http://localhost:8501`).

## Run as a web app using Docker

```bash
docker build -t textile-ai-dashboard .
docker run -p 8501:8501 textile-ai-dashboard
```

Open: `http://localhost:8501`

## Deploy online

You can deploy this dashboard directly on platforms like:
- Streamlit Community Cloud
- Render
- Railway
- Azure App Service
- AWS ECS/Fargate

Main app entrypoint: `dashboard.py`

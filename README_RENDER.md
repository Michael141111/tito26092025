# Deploy Tito Backend on Render — Step-by-step

## 0) Prepare your project locally
- Put all portal HTML files inside `data_html/` (same level as `app.py`).
- Confirm Python 3.11+.

## 1) Create a GitHub repository
```bash
# from the folder that contains app.py, requirements.txt, render.yaml ...
git init
git add .
git commit -m "Tito backend initial"
# create a repo on GitHub (e.g., tito-backend) then run:
git branch -M main
git remote add origin https://github.com/<YOUR-USER>/tito-backend.git
git push -u origin main
```

## 2) Create Render service
- Go to https://dashboard.render.com
- Click **New** → **Web Service** → **Connect** your GitHub repo
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Click **Create Web Service**

## 3) Set environment variables
In **Settings → Environment**:
- `OPENAI_API_KEY = <your key>`
- `CORS_ORIGINS = https://adrc-p.com,https://www.adrc-p.com`
- `BASE_URL_STATIC_HTML = /html`

## 4) First boot & indexing
- After first deploy, open **Logs** and wait until the server is live.
- The app has a startup hook that **auto-ingests** if no index exists.
- (Optional) Manually trigger:
  - `POST https://<your-service>.onrender.com/ingest`

## 5) Health & Test
- `GET https://<your-service>.onrender.com/api/health` → should return OK
- `POST https://<your-service>.onrender.com/api/rag` with:
```json
{ "query": "What is the Credit Policy scope?", "lang": "en" }
```

## 6) Connect Cloudflare Worker
Set your Worker env var:
- `RAG_BACKEND_ORIGIN = https://<your-service>.onrender.com`

Routes:
- `adrc-p.com/api/*`      → Worker
- `www.adrc-p.com/api/*`  → Worker

Your frontend calls `/api/rag` on your domain; Worker proxies to Render backend.

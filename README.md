# EstateIQ — AI Real Estate Intelligence

Portfolio starter combining Python, data analysis, ML price prediction, property recommendations, Generative AI, RAG, FastAPI and a web UI.

## Run on Ubuntu
```bash
cd ai_real_estate_intelligence
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```
Optional: edit `.env` and set `GROQ_API_KEY` from https://console.groq.com/. Without a key, document retrieval works and the UI explains that generation is disabled. First RAG use downloads `sentence-transformers/all-MiniLM-L6-v2`.

```bash
uvicorn app:app --reload --port 8050
```
Open http://127.0.0.1:8050

## Core files
- `app.py`: FastAPI routes and request validation.
- `services/ml_service.py`: pandas analysis, Random Forest model, metrics, similarity-based recommendations.
- `services/rag_service.py`: document chunking, MiniLM embeddings via ChromaDB, top-3 retrieval, optional Groq LLM.
- `knowledge/`: starter knowledge notes indexed into the RAG collection.
- `data/properties.csv`: 120 synthetic demo listings.

## GitHub
```bash
git init
git add .
git commit -m "Build EstateIQ AI real estate intelligence"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-real-estate-intelligence.git
git push -u origin main
```
Never commit `.env` or API keys.

## Important limitation
The supplied property dataset is synthetic and only demonstrates the pipeline. Do not use these prices for real decisions. Replace it with verified local transaction/listing data, then retrain and evaluate the model before describing predictions as market estimates.

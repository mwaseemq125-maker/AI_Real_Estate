from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from services.ml_service import stats, load_data, estimate, recommend
from services.rag_service import ask, index_file, retrieve

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="EstateIQ AI Real Estate Intelligence")
app.mount("/static", StaticFiles(directory=str(ROOT/"static")), name="static")
templates = Jinja2Templates(directory=str(ROOT/"templates"))

class PropertyInput(BaseModel):
    area_sqft: float = Field(gt=100, le=100000)
    bedrooms: int = Field(ge=0, le=20)
    bathrooms: int = Field(ge=0, le=20)
    property_age: int = Field(default=0, ge=0, le=200)
    city: str
    neighborhood: str
    property_type: str

class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"s": stats()})

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/api/properties")
def properties(city: str="", property_type: str=""):
    df = load_data()
    if city: df = df[df.city.str.lower()==city.lower()]
    if property_type: df = df[df.property_type.str.lower()==property_type.lower()]
    return df.to_dict("records")

@app.post("/api/predict")
def predict(data: PropertyInput):
    try: return estimate(data.model_dump())
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/recommend")
def recommendations(data: PropertyInput):
    return {"recommendations":recommend(data.model_dump())}

@app.post("/api/chat")
def chat(data: Question):
    try: return ask(data.question)
    except Exception as e: raise HTTPException(500, f"Assistant error: {e}")

@app.get("/api/search")
def semantic_search(q: str):
    if not q.strip(): raise HTTPException(400,"Enter a search query.")
    return {"results":retrieve(q,3)}

@app.post("/api/upload")
async def upload(file: UploadFile=File(...)):
    raw = await file.read()
    if len(raw)>10*1024*1024: raise HTTPException(413,"Maximum upload size is 10 MB.")
    try: return {"message":"Document indexed","chunks_indexed":index_file(file.filename or "upload.txt",raw)}
    except ValueError as e: raise HTTPException(400,str(e))

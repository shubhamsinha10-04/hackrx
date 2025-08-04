import os
import time
from typing import List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from utils import answer_questions

load_dotenv()

app = FastAPI(
    title="HackRx Insurance Q&A System",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

security = HTTPBearer()
TEAM_TOKEN = os.getenv("TEAM_TOKEN")
if not TEAM_TOKEN:
    raise RuntimeError("TEAM_TOKEN is not set in .env")

class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(payload: QueryRequest, token: str = Depends(verify_token)):
    start = time.time()
    answers = await answer_questions(payload.questions)
    duration = time.time() - start
    print(f"[API] Processed request in {duration:.2f}s")
    return QueryResponse(answers=answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)




from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram
import time

app = FastAPI()

# Monitoring metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total RAG requests')
REQUEST_LATENCY = Histogram('rag_request_latency_seconds', 'RAG request latency')

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    latency_ms: float


# Load models at startup
@app.on_event("startup")
async def load_models():
    global rag_pipeline, embeddingss

    #iniitialize rag componsents here
    pass

@app.post("/query", response_model=QueryResponse):
async def query_rag(request: QueryRequest):
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        
        # Process query
        result = rag_pipeline.query(request.questiion, top_k=request.top_k)

        latency = (time.time() - start_time) * 1000
        REQUEST_LATENCY.observe(latency / 1000)

        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result.get('confidence', 0.0),
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Docker deployment
'''
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host]", "0.0.0.0", "--port", 8000"]

'''




# Kubernetes deployment
'''
apiVersion: apps/v1
kind: Deployment
metadata:
    name: rag-service
spec:
    replicas: 3
    selector:
        matchLabels
            app: rag-service
    template:
        metadata:
            labels:
                app: rag-service
        spec:
            containers:
            - name: rag
              image: your-registry/rag-service:latest
              resources:
                requests:
                    memory: "4Gi"
                    cpu: "2"
                    nvidia.com/gpu: 1
                limits:
                    memory: "8Gi"
                    cpu: "4"
                    nvidia.com/gpu: 1

'''
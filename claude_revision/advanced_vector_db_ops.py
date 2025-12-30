import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize Qdrant
client = qdrant_client.QdrantClient(host="localhost", port=6333)

# create collections with payloaded indexing
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536,distance=Distance.COSINE),
)

# batch upsert for scale
from typing import List
import numpy as np

def batch_upsert(texts: List[str], embeddings: np.ndarray, metadata: List[dict]):
    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={"text": text, **meta}
        )

        for idx, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata))
    ]

    client.upsert(
        collection_name="documents",
        points=points,
        wait=True
    )


# Hybrid search: dense + sparse

def hybrid_search(query_vector: List[float], query_text: str. top_k: int = 10):
    # dense vector search
    dense_results = client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

    # keyword filtering
    filtered_results = client.search(
        collection_name="documents",
        query_vector=query_vector,
        query_filter={
            "must": [
                {"key": "text", "match" : {"text": query_text}}
            ]
        },
        limit=top_k
    )

    # combine and rerank
    return dense_results, filtered_results
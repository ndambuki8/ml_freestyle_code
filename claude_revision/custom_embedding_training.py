import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#prepare domain-specific training data
train_examples = [
    InputExample(texts=['query text', 'relevant document'], label=1.0),
    InputExample(texts=['query text', 'irrelevant document'], label=0.0)
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Fine-tune with contrastive loss
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100,
    output_path='./custom-embeddings'
)

# evaluateee retrieval performance
from sentence_transformers.evaluation import InformationRetrievalEvaluator

queries = {'q1': "What is machine learning?"}
corpus = {'doc1': 'ML is a subset of AI....', 'doc2':"Cooking recipes..."}
relevant_docs = {'q1':{'doc1'}}

evaluator = InformationRetrievalEvaluator(queries, corus, relevant_docs)
results = evaluator(model)
print(f"nDCG@10: {results}")

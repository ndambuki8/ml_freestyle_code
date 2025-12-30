from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, util

class RAGEvaluator:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def faithfullness_score(self, answer:str, context: str) -> float:
        """Measure if answer is grounded in context"""
        answer_emb = self.embedding_model.encode(answer)
        context_emb = self.embedding_model.encode(context)
        return float(util.cos_sim(answer_emb, context_emb))
    
    def answer_relevancy(self, answer: str, question: str) -> float:
        """Meausure if answer addresses the question"""
        answer_emb = self.embedding_model.encode(answer)
        question_emb = self.embedding_model.encode(question)
        return float(util.cos_sim(answer_emb, question_emb))
    
    def context_precision(self, contexts:List[str], ground_truth: str) -> float:
        """Measure if retrieved contexts are relevant"""
        gt_emb = self.embedding_model.encode(ground_truth)
        context_embs = self.embedding_model.encode(contexts)
        similarities = util.cos_sim(gt_emb, context_embs)[0]
        return float(similarities.mean())
    
    def hallucination_detection(self, answer: str, context: str, llm) -> Dict:
        """Use LLM to detect hallucinations"""
        prompt = f"""Check if this answer contains information not present in the context.

        Context: {context}
        Answer: {answer}

        Does the answer contain hallucinations? reply Yes or No and explain."""
        result = llm.predict(prompt)
        return {"verdict": result, "has_halucination": "Yes" in result.upper()}
    
    def evaluate_pipelines(self, question: str, answer: str,
                           contexts: List[str], ground_truth: str) -> Dict:
        context_str = " ".join(contexts)
    
        return {
            "faithfullness": self.faithfullness_score(answer, context_str),
            "relevancy": self.answer_relevancy(answer, question),
            "context_precision": self.context_precision(contexts, ground_truth),
            "answer_length": len(answer.split())
        }
    

# Usage with test dataset

evaluator = RAGEvaluator()

test_cases = [
    {
        "question": "What is  RAG?",
        "answer": "RAG stands for retrieval augmented generation...",
        "contexts": ["RAG is a technique..", "It combines retrieval...."],
        "ground_truth":"Retrieval Augmented Generated combines..."
    }
]

results = [evaluator.evaluate_pipelines(**case) for case in test_cases]
print(f"Average Faithfullness: {np.mean([r['faithfullness'] for r in results])}")
from typing import Dict, List, Tuple, Optional
import asyncio
import logging
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from ..utils.resources import ResourceManager

logger = logging.getLogger(__name__)

class ModelOrchestrator:
    """Manages multiple LLM models and their interactions."""
    
    def __init__(self):
        self.models = {
            "mistral": OllamaLLM(model="mistral:latest"),
            "llama": OllamaLLM(model="llama3.2:latest"),
            "granite": OllamaLLM(model="granite3.1-dense:8b")
        }
        
    async def get_response(self, model_name: str, prompt: str) -> str:
        try:
            return await self.models[model_name].ainvoke(prompt)
        except Exception as e:
            logger.error(f"Error with model {model_name}: {e}")
            return ""

class ContextManager:
    """Manages document context and retrieval."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.vectorstore = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    async def initialize(self, documents: List[Document]):
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.vectorstore = Chroma(
            collection_name="pdf_content",
            embedding_function=embeddings
        )
        self.vectorstore.add_documents(documents)
        
    async def get_relevant_context(self, question: str, 
                                 k: int = 5, 
                                 fetch_k: int = 10) -> Tuple[str, List[Dict]]:
        if not self.vectorstore:
            return "", []
            
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                'k': k,
                'fetch_k': fetch_k,
                'lambda_mult': 0.7,
                'filter': self._get_chapter_filter(question)
            }
        )
        
        docs = await retriever.ainvoke(question)
        return self._process_retrieved_docs(docs)
        
    def _get_chapter_filter(self, question: str) -> Optional[Dict]:
        import re
        chapter_match = re.search(r'chapter\s+(\d+)', question.lower())
        return {"chapter_number": chapter_match.group(1)} if chapter_match else None
        
    def _process_retrieved_docs(self, docs: List[Document]) -> Tuple[str, List[Dict]]:
        chapter_groups = {}
        for doc in docs:
            chapter_key = f"{doc.metadata['chapter_number']}_{doc.metadata['chapter_title']}"
            if chapter_key not in chapter_groups:
                chapter_groups[chapter_key] = []
            chapter_groups[chapter_key].append(doc)

        context_parts = []
        metadata_list = []
        
        for chapter_key, chapter_docs in chapter_groups.items():
            chapter_docs.sort(key=lambda x: x.metadata['paragraph_index'])
            context_parts.append(f"\n=== {chapter_docs[0].metadata['chapter_context']} ===\n")
            
            for doc in chapter_docs:
                context_parts.append(doc.page_content)
                metadata_list.append(doc.metadata)

        return "\n".join(context_parts), metadata_list

class ResponseGenerator:
    """Generates and combines model responses."""
    
    def __init__(self):
        self.response_cache = {}
        
    def _create_prompt(self, model_name: str, question: str, context: str) -> str:
        prompts = {
            "mistral": f"""You are a precise and factual assistant. Answer based only on the provided context.
Context: {context}
Question: {question}
Requirements:
- Use only information from the context
- Be specific and cite relevant parts
- Say "I don't know" if the context lacks necessary information
- Focus on accuracy over speculation""",

            "llama": f"""You are an analytical assistant specializing in complex reasoning.
Context: {context}
Question: {question}
Requirements:
- Analyze relationships between concepts
- Explain underlying principles
- Support claims with context evidence
- Be explicit about assumptions""",

            "granite": f"""You are a practical synthesizer of information.
Context: {context}
Question: {question}
Requirements:
- Summarize key points clearly
- Use concrete examples
- Highlight practical applications
- Focus on main themes"""
        }
        return prompts.get(model_name, prompts["mistral"])
        
    def _get_weights(self, question: str, context_metadata: List[Dict]) -> Dict[str, float]:
        import re
        
        weights = {"mistral": 0.33, "llama": 0.33, "granite": 0.34}
        
        patterns = {
            "factual": r'what|when|where|who|how many',
            "analytical": r'why|how|explain|analyze|compare',
            "summary": r'summarize|overview|brief|main points'
        }
        
        for pattern_type, pattern in patterns.items():
            if re.search(pattern, question.lower()):
                if pattern_type == "factual":
                    weights["mistral"] += 0.1
                    weights["llama"] -= 0.05
                    weights["granite"] -= 0.05
                elif pattern_type == "analytical":
                    weights["llama"] += 0.1
                    weights["mistral"] -= 0.05
                    weights["granite"] -= 0.05
                else:  # summary
                    weights["granite"] += 0.1
                    weights["mistral"] -= 0.05
                    weights["llama"] -= 0.05
                    
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
        
    def combine_responses(self, responses: Dict[str, str], weights: Dict[str, float]) -> str:
        valid_responses = {k: v for k, v in responses.items() if v}
        if not valid_responses:
            return "Unable to generate a response."
            
        # Convert to sentences and calculate similarities
        all_sentences = {
            model: [s.strip() + '.' for s in response.split('.') if len(s.strip()) > 20]
            for model, response in valid_responses.items()
        }
        
        vectorizer = TfidfVectorizer()
        all_vectors = []
        for sentences in all_sentences.values():
            if sentences:
                vectors = vectorizer.fit_transform(sentences)
                all_vectors.extend(vectors.toarray())
                
        if not all_vectors:
            return list(valid_responses.values())[0]
            
        similarity_matrix = cosine_similarity(all_vectors)
        
        # Score and select sentences
        scored_sentences = {}
        for model, sentences in all_sentences.items():
            for i, sentence in enumerate(sentences):
                consensus_score = np.mean(similarity_matrix[i])
                final_score = consensus_score * weights[model]
                
                key = sentence.lower()
                if key not in scored_sentences or final_score > scored_sentences[key][0]:
                    scored_sentences[key] = (final_score, sentence)
                    
        sorted_sentences = sorted(scored_sentences.values(), key=lambda x: x[0], reverse=True)
        return ' '.join(s[1] for s in sorted_sentences[:5])

class MultiModelChat:
    """Main chat class orchestrating the entire conversation flow."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.orchestrator = ModelOrchestrator()
        self.context_manager = ContextManager()
        self.response_generator = ResponseGenerator()
        
    async def initialize(self, documents: List[Document]):
        await self.context_manager.initialize(documents)
        
    async def get_response(self, question: str) -> Dict:
        try:
            # Get context
            context, context_metadata = await self.context_manager.get_relevant_context(question)
            
            # Get model weights
            weights = self.response_generator._get_weights(question, context_metadata)
            
            # Generate responses
            tasks = []
            for model_name in self.orchestrator.models:
                prompt = self.response_generator._create_prompt(model_name, question, context)
                tasks.append(self.orchestrator.get_response(model_name, prompt))
                
            responses = await asyncio.gather(*tasks)
            
            # Combine responses
            model_responses = dict(zip(self.orchestrator.models.keys(), responses))
            combined = self.response_generator.combine_responses(model_responses, weights)
            
            return {
                "combined_response": combined,
                "individual_responses": model_responses,
                "weights_used": weights
            }
            
        except Exception as e:
            logger.error(f"Error in chat response: {e}")
            raise
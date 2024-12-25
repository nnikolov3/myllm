from typing import Dict, List, Tuple, Optional
import asyncio
import logging
from datetime import datetime
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from ..utils.resources import ResourceManager
from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class ModelNotFoundError(ModelError):
    """Exception raised when a model is not available."""
    pass

class ModelTimeoutError(ModelError):
    """Exception raised when a model request times out."""
    pass

class DocumentProcessingError(Exception):
    """Exception raised when document processing fails."""
    pass

class ConversationHistory:
    """Manages conversation history."""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_interaction(self, question: str, response: Dict):
        """Add an interaction to history."""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_recent_context(self, n: int = 3) -> str:
        """Get recent conversation context."""
        recent = self.history[-n:] if len(self.history) > n else self.history
        context = []
        for item in recent:
            context.append(f"Q: {item['question']}")
            context.append(f"A: {item['response']['combined_response']}")
        return "\n".join(context)

    def clear(self):
        """Clear conversation history."""
        self.history.clear()

class ModelOrchestrator:
    """Manages multiple LLM models and their interactions."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration."""
        self.config_manager = config_manager
        self.models = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize models based on configuration."""
        model_configs = self.config_manager.get_active_models()
        
        for model_name, config in model_configs.items():
            try:
                self.models[model_name] = OllamaLLM(
                    model=config.name,
                    timeout=config.timeout
                )
                logger.info(f"Initialized model: {model_name} ({config.name})")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {e}")
    
    async def get_response(self, model_name: str, prompt: str) -> str:
        """Get response from specified model with error handling."""
        if model_name not in self.models:
            raise ModelNotFoundError(f"Model {model_name} not found")
            
        try:
            return await self.models[model_name].ainvoke(prompt)
        except asyncio.TimeoutError:
            raise ModelTimeoutError(f"Request to {model_name} timed out")
        except Exception as e:
            logger.error(f"Error with model {model_name}: {e}")
            return ""

    async def check_models_health(self) -> Dict[str, bool]:
        """Check health status of all models."""
        health_status = {}
        test_prompt = "Test prompt for health check."
        
        for model_name in self.models:
            try:
                await self.get_response(model_name, test_prompt)
                health_status[model_name] = True
            except Exception as e:
                logger.error(f"Health check failed for {model_name}: {e}")
                health_status[model_name] = False
                
        return health_status

class ContextManager:
    """Manages document context and retrieval."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize context manager."""
        self.vectorstore = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._cache = {}

    async def initialize(self, documents: List[Document]):
        """Initialize the context manager with documents."""
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            self.vectorstore = Chroma(
                collection_name="pdf_content",
                embedding_function=embeddings
            )
            
            # Process and add documents
            processed_docs = self._process_documents(documents)
            if processed_docs:
                self.vectorstore.add_documents(processed_docs)
                logger.info(f"Added {len(processed_docs)} documents to vectorstore")
            else:
                raise ValueError("No valid documents to process")

        except Exception as e:
            logger.error(f"Error initializing context manager: {e}")
            raise

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents for vectorstore."""
        processed = []
        for doc in documents:
            try:
                # Filter metadata to basic types
                filtered_metadata = {
                    k: v for k, v in doc.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
                processed.append(Document(
                    page_content=doc.page_content,
                    metadata=filtered_metadata
                ))
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                
        return processed

    @lru_cache(maxsize=1000)
    async def get_relevant_context(self, 
                                question: str, 
                                k: int = 5) -> Tuple[str, List[Dict]]:
        """Get relevant context for a question with caching."""
        if not self.vectorstore:
            return "", []

        try:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': k,
                    'fetch_k': k * 2,
                    'lambda_mult': 0.7
                }
            )
            docs = await retriever.ainvoke(question)
            return self._process_retrieved_docs(docs)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "", []

    def _process_retrieved_docs(self, docs: List[Document]) -> Tuple[str, List[Dict]]:
        """Process retrieved documents into context."""
        try:
            context_parts = []
            metadata_list = []
            
            for doc in docs:
                if isinstance(doc, Document):
                    context_parts.append(doc.page_content)
                    metadata_list.append(doc.metadata)

            return "\n".join(context_parts), metadata_list
            
        except Exception as e:
            logger.error(f"Error processing retrieved documents: {e}")
            return "", []

    def clear_cache(self):
        """Clear the context cache."""
        self.get_relevant_context.cache_clear()

class ResponseGenerator:
    """Generates and combines model responses."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration."""
        self.config_manager = config_manager
        self.cache = {}
        
    def _create_prompt(self, model_name: str, question: str, context: str) -> str:
        """Create appropriate prompt for each model."""
        base_prompt = f"""Answer based on the provided context.
Context: {context}
Question: {question}
Requirements:
- Use only information from the context
- Be specific and cite relevant parts
- Say "I don't know" if the context lacks necessary information"""

        model_config = self.config_manager.get_model_config(model_name)
        if model_config:
            # Access model_config attributes directly instead of using .get()
            return base_prompt  # Since ModelConfig doesn't have instructions field, just return base prompt
        return base_prompt
        
    def combine_responses(self, 
                         responses: Dict[str, str], 
                         weights: Dict[str, float]) -> str:
        """Combine model responses using weighted approach."""
        if not responses:
            return "Unable to generate a response."
            
        # Process responses
        all_sentences = self._split_into_sentences(responses)
        if not all_sentences:
            return list(responses.values())[0]
            
        # Calculate similarities and scores
        scores = self._calculate_sentence_scores(all_sentences, weights)
        
        # Select and combine best sentences
        return self._combine_best_sentences(scores)

    def _split_into_sentences(self, responses: Dict[str, str]) -> Dict[str, List[str]]:
        """Split responses into sentences."""
        return {
            model: [s.strip() + '.' for s in response.split('.') if len(s.strip()) > 20]
            for model, response in responses.items() if response
        }

    def _calculate_sentence_scores(self, 
                                 sentences: Dict[str, List[str]], 
                                 weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate scores for sentences."""
        vectorizer = TfidfVectorizer()
        all_sentences = []
        sentence_map = {}
        
        for model, model_sentences in sentences.items():
            for sentence in model_sentences:
                if sentence not in sentence_map:
                    sentence_map[sentence] = []
                sentence_map[sentence].append((model, len(all_sentences)))
                all_sentences.append(sentence)
                
        if not all_sentences:
            return {}
            
        vectors = vectorizer.fit_transform(all_sentences)
        similarity_matrix = cosine_similarity(vectors)
        
        scores = {}
        for sentence, occurrences in sentence_map.items():
            total_score = 0
            for model, idx in occurrences:
                consensus_score = np.mean(similarity_matrix[idx])
                total_score += consensus_score * weights[model]
            scores[sentence] = total_score / len(occurrences)
            
        return scores

    def _combine_best_sentences(self, scores: Dict[str, float], limit: int = 5) -> str:
        """Combine best sentences into coherent response."""
        best_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return ' '.join(sentence for sentence, _ in best_sentences)

class MultiModelChat:
    """Main chat class orchestrating the entire conversation flow."""
    
    def __init__(self, 
                 resource_manager: ResourceManager,
                 config_manager: ConfigManager):
        """Initialize chat system."""
        self.resource_manager = resource_manager
        self.config_manager = config_manager
        self.orchestrator = ModelOrchestrator(config_manager)
        self.context_manager = ContextManager()
        self.response_generator = ResponseGenerator(config_manager)
        self.conversation = ConversationHistory(
            max_history=config_manager.get_system_config().cache_size
        )
        
    async def initialize(self, documents: List[Document]):
        """Initialize chat with documents."""
        await self.context_manager.initialize(documents)
        await self._check_system_health()
        
    async def _check_system_health(self):
        """Check system health status."""
        health_status = await self.orchestrator.check_models_health()
        if not any(health_status.values()):
            raise ModelError("No models are available")
        logger.info(f"System health check: {health_status}")
        
    async def get_response(self, question: str) -> Dict:
        """Get response for a question."""
        try:
            # Get context and conversation history
            context, context_metadata = await self.context_manager.get_relevant_context(question)
            conversation_context = self.conversation.get_recent_context()
            
            # Combine contexts
            full_context = f"{conversation_context}\n\n{context}" if conversation_context else context
            
            # Generate responses from all healthy models
            tasks = []
            for model_name, config in self.config_manager.get_active_models().items():
                prompt = self.response_generator._create_prompt(model_name, question, full_context)
                tasks.append(self.orchestrator.get_response(model_name, prompt))
                
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and create response dictionary
            model_responses = {}
            for model_name, response in zip(self.config_manager.get_active_models(), responses):
                if not isinstance(response, Exception):
                    model_responses[model_name] = response
            
            if not model_responses:
                return {
                    "combined_response": "All models failed to generate a response.",
                    "individual_responses": {},
                    "weights_used": {}
                }
                
            # Get model weights from configuration
            weights = {
                name: config.weight 
                for name, config in self.config_manager.get_active_models().items()
                if name in model_responses
            }
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Combine responses
            combined = self.response_generator.combine_responses(model_responses, weights)
            
            response_data = {
                "combined_response": combined,
                "individual_responses": model_responses,
                "weights_used": weights
            }
            
            # Add to conversation history
            self.conversation.add_interaction(question, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in chat response: {e}")
            raise

    def clear_caches(self):
        """Clear all internal caches."""
        self.context_manager.clear_cache()
        self.response_generator.cache.clear()
        self.conversation.clear()

    async def refresh_context(self, documents: List[Document]):
        """Refresh the context with new documents."""
        try:
            await self.context_manager.initialize(documents)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to refresh context: {e}")

    async def shutdown(self):
        """Cleanup and shutdown chat system."""
        try:
            self.clear_caches()
            await self.resource_manager.cleanup()
            logger.info("Chat system shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
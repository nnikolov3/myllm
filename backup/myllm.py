from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text
from rich.padding import Padding
import textwrap
import time

import os
import sys
import asyncio
import logging
import pickle
import numpy as np
import torch
import psutil
import gc
from datetime import datetime
from typing import List, Dict, Optional, Generator
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Initialize Rich console with specific width
console = Console(width=80)

class UserInterface:
    def __init__(self):
        self.console = Console(width=80)
        
    def startup_banner(self):
        """Display startup banner."""
        console.print(
            Panel.fit(
                "🤖 PDF Chat System\n[dim]Type 'exit' to quit, 'help' for commands[/dim]",
                border_style="blue",
                width=80
            )
        )
        
    def show_progress(self):
        """Create and return a progress context."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            width=80
        )
    
    def format_text(self, text: str) -> str:
        """Format text with proper wrapping and spacing."""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Format each paragraph
        formatted_paragraphs = []
        for para in paragraphs:
            # Wrap text at 75 chars to account for margins
            wrapped = textwrap.fill(para.strip(), width=75)
            formatted_paragraphs.append(wrapped)
        
        # Join with double newlines for paragraph spacing
        return '\n\n'.join(formatted_paragraphs)
    
    def display_response(self, response: str, model_responses: Optional[Dict] = None):
        """Display the response in a nicely formatted way."""
        # Format the main response
        formatted_response = self.format_text(response)
        
        # Create styled text with proper spacing
        styled_response = Group(
            Text(''),  # Top padding
            Markdown(formatted_response),
            Text('')   # Bottom padding
        )
        
        # Display main response in a panel
        console.print(
            Panel(
                styled_response,
                title="Response",
                border_style="green",
                width=80,
                padding=(0, 2)
            )
        )
        
        # Show model details if available
        if model_responses and Prompt.ask(
            "Show individual model responses?", 
            choices=["y", "n"], 
            default="n"
        ) == "y":
            for model, resp in model_responses.items():
                if resp:
                    formatted_model_response = self.format_text(resp)
                    console.print(
                        Panel(
                            Group(
                                Text(''),
                                Markdown(formatted_model_response),
                                Text('')
                            ),
                            title=f"Model: {model}",
                            border_style="blue",
                            width=80,
                            padding=(0, 2)
                        )
                    )
                    console.print()  # Add spacing between model responses
    
    def get_input(self) -> str:
        """Get user input in a styled way."""
        return Prompt.ask("\n[bold blue]Ask a question[/bold blue]")
        
    def show_error(self, error_msg: str):
        """Display error message."""
        console.print(
            Panel(
                self.format_text(error_msg),
                title="Error",
                border_style="red",
                width=80
            )
        )

    def show_help(self):
        """Display help information."""
        help_text = """
        Available Commands:
        - exit: Quit the program
        - help: Show this help message
        - clear: Clear the screen
        """
        console.print(
            Panel(
                self.format_text(help_text),
                title="Help",
                border_style="blue",
                width=80
            )
                )
                


# Configure logging based on debug flag
def setup_logging(debug_mode: bool = False):
    """Setup logging configuration based on debug mode."""
    handlers = []
    
    # Always log to file with DEBUG level
    file_handler = logging.FileHandler('debug.log')
    file_handler.setLevel(logging.DEBUG)
    handlers.append(file_handler)
    
    # Console handler with level based on debug mode
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Only show logger name and message in console for cleaner output
    console_handler.setFormatter(logging.Formatter(
        '%(message)s' if not debug_mode else '%(asctime)s - %(levelname)s - %(message)s'
    ))
    handlers.append(console_handler)
    
    # Set up basic configuration
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    cpu_count: int
    total_memory: int
    gpu_available: bool
    gpu_count: int

class ResourceManager:
    def __init__(self):
        self._lock = Lock()
        self.resources = self._initialize_resources()
        self._log_system_info()

    def _initialize_resources(self) -> SystemResources:
        return SystemResources(
            cpu_count=psutil.cpu_count(logical=True),
            total_memory=psutil.virtual_memory().total,
            gpu_available=torch.cuda.is_available(),
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0
        )

    def _log_system_info(self) -> None:
        logger.info("System Resources:")
        logger.info(f"CPU Count: {self.resources.cpu_count}")
        logger.info(f"Memory: {self.resources.total_memory / (1024**3):.2f} GB")
        logger.info(f"GPU Available: {self.resources.gpu_available}")
        if self.resources.gpu_available:
            logger.info(f"GPU Count: {self.resources.gpu_count}")
            for i in range(self.resources.gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    def optimize_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        with self._lock:
            if self.resources.gpu_available:
                available_memory = int(torch.cuda.get_device_properties(0).total_memory * 0.8)
            else:
                available_memory = int(psutil.virtual_memory().available * 0.8)
            # Estimate memory per document (adjust based on your needs)
            estimated_memory_per_doc = 1024 * 1024  # 1MB per document
            return max(1, available_memory // estimated_memory_per_doc)

    async def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if self.resources.gpu_available:
            with self._lock:
                torch.cuda.empty_cache()
                gc.collect()
                await asyncio.sleep(0)

class PDFProcessor:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.thread_count = max(1, self.resource_manager.resources.cpu_count - 1)
        self.batch_size = self.resource_manager.optimize_batch_size()

    def load_pdf_lazy(self, pdf_path: str) -> Generator[Document, None, None]:
        """Lazily load a PDF document."""
        logger.debug(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        yield from loader.load()

    async def process_documents(self, docs_generator: Generator[Document, None, None]) -> List[Document]:
        """Process documents in optimized batches."""
        logger.debug("Starting document processing")
        processed_docs = []
        current_batch = []

        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            async for doc in self._aiter(docs_generator):
                current_batch.append(doc)
                
                if len(current_batch) >= self.batch_size:
                    processed = await self._process_batch(current_batch, executor)
                    processed_docs.extend(processed)
                    current_batch = []
                    await self.resource_manager.clear_gpu_memory()

            if current_batch:  # Process remaining documents
                processed = await self._process_batch(current_batch, executor)
                processed_docs.extend(processed)

        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs

    async def _process_batch(self, batch: List[Document], executor: ThreadPoolExecutor) -> List[Document]:
        """Process a batch of documents in parallel."""
        loop = asyncio.get_event_loop()
        sub_batches = np.array_split(batch, min(len(batch), self.thread_count))
        
        tasks = [
            loop.run_in_executor(executor, self._process_sub_batch, sub_batch)
            for sub_batch in sub_batches
        ]
        
        results = await asyncio.gather(*tasks)
        return [doc for batch_result in results for doc in batch_result]

    def _process_sub_batch(self, docs: List[Document]) -> List[Document]:
        """Process a sub-batch of documents."""
        processed_docs = []
        for doc in docs:
            try:
                processed_docs.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "processed_timestamp": datetime.now().isoformat()
                    }
                ))
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        return processed_docs

    @staticmethod
    async def _aiter(generator):
        """Convert a regular generator to an async generator."""
        for item in generator:
            yield item

class MultiModelChat:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.models = {
            "mistral": OllamaLLM(model="mistral:latest"),
            "llama": OllamaLLM(model="llama3.2:latest"),
            "granite": OllamaLLM(model="granite3.1-dense:8b")
        }
        self.response_cache = {}
        self.vectorstore = None

    async def initialize_vectorstore(self, documents: List[Document]):
        """Initialize the vector store with processed documents."""
        logger.debug("Initializing vector store")
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            self.vectorstore = Chroma(
                collection_name="pdf_content",
                embedding_function=embeddings
            )
            self.vectorstore.add_documents(documents)
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    @lru_cache(maxsize=100)
    def _get_cached_response(self, model_name: str, question: str) -> Optional[str]:
        """Get cached response if available."""
        cache_key = f"{model_name}:{question}"
        return self.response_cache.get(cache_key)

    async def get_response(self, question: str) -> Dict:
        """Get responses from all models in parallel."""
        logger.debug(f"Processing question: {question}")
        
        try:
            # Get relevant context if vectorstore exists
            context = ""
            if self.vectorstore:
                retriever = self.vectorstore.as_retriever()
                chunks = await retriever.ainvoke(question)
                context = "\n".join(doc.page_content for doc in chunks)

            # Check cache first
            cached_responses = {
                name: self._get_cached_response(name, question)
                for name in self.models.keys()
            }
            
            # Generate missing responses in parallel
            tasks = []
            for name, model in self.models.items():
                if cached_responses[name] is None:
                    tasks.append(self._get_model_response(name, model, question, context))

            if tasks:
                new_responses = await asyncio.gather(*tasks)
                # Update cache with new responses
                for name, response in zip(self.models.keys(), new_responses):
                    if response:
                        self.response_cache[f"{name}:{question}"] = response

            # Combine all responses
            all_responses = {
                name: cached_responses[name] or self.response_cache.get(f"{name}:{question}", "")
                for name in self.models.keys()
            }

            # Create combined response
            combined = self._combine_responses(all_responses)
            
            return {
                "combined_response": combined,
                "individual_responses": all_responses
            }

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            raise

    async def _get_model_response(self, name: str, model: OllamaLLM, question: str, context: str) -> str:
        """Get response from a single model."""
        try:
            # Specialized prompts for each model
            prompts = {
                "mistral": f"""You are a precise and thorough assistant. Using the provided context, answer the question accurately and comprehensively. If the context doesn't contain enough information, say so.

Context: {context}

Question: {question}

Provide a detailed and accurate response that directly addresses the question. Include specific examples or citations from the context when relevant.""",

                "llama": f"""You are an analytical assistant focusing on technical accuracy. Based on the provided context, answer the question with precision and depth.

Context: {context}

Question: {question}

Analyze the available information carefully and provide a well-structured response. Make connections between different parts of the context when relevant. If certain aspects are unclear or not covered in the context, acknowledge these limitations.""",

                "granite": f"""You are a practical and precise assistant. Using the provided context, answer the question with concrete examples and clear explanations.

Context: {context}

Question: {question}

Provide a clear, practical response that emphasizes accuracy and relevance. Use specific examples from the context to support your points. If the context doesn't fully address the question, be explicit about what information is missing."""
            }

            # Get the appropriate prompt for this model
            prompt = prompts.get(name, prompts["mistral"])  # Default to mistral prompt if model not found
            
            response = await model.ainvoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error with model {name}: {e}")
            return ""

    def _combine_responses(self, responses: Dict[str, str]) -> str:
        """Combine responses from different models intelligently."""
        # Filter out empty responses
        valid_responses = {k: v for k, v in responses.items() if v}
        
        if not valid_responses:
            return "I couldn't generate a proper response."

        # Model weights based on their strengths
        model_weights = {
            "mistral": 0.4,  # Good for general knowledge and clarity
            "llama": 0.35,   # Strong in technical accuracy
            "granite": 0.25  # Good for practical examples
        }

        # Split responses into sentences and remove duplicates
        all_sentences = {}
        for model, response in valid_responses.items():
            sentences = [s.strip() + '.' for s in response.replace('\n', ' ').split('.') if s.strip()]
            all_sentences[model] = sentences

        # Score and combine unique information
        scored_sentences = {}
        for model, sentences in all_sentences.items():
            for sentence in sentences:
                # Skip very short sentences
                if len(sentence.split()) < 4:
                    continue
                    
                # Calculate sentence score based on:
                # 1. Model weight
                # 2. Sentence length (prefer medium-length sentences)
                # 3. Information density (ratio of unique words to total words)
                words = sentence.split()
                unique_words = len(set(words))
                length_score = min(1.0, len(words) / 20)  # Prefer sentences around 20 words
                info_density = unique_words / len(words)
                
                score = (
                    model_weights[model] * 
                    length_score * 
                    info_density
                )
                
                # Use sentence as key to avoid duplicates
                key = sentence.lower()
                if key not in scored_sentences or score > scored_sentences[key][0]:
                    scored_sentences[key] = (score, sentence)

        # Sort sentences by score and combine
        sorted_sentences = sorted(
            scored_sentences.values(),
            key=lambda x: x[0],
            reverse=True
        )

        # Combine sentences into coherent paragraphs
        paragraphs = []
        current_paragraph = []
        
        for _, sentence in sorted_sentences:
            current_paragraph.append(sentence)
            
            # Start new paragraph every few sentences
            if len(current_paragraph) >= 3:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))

        # Join paragraphs with newlines
        return '\n\n'.join(paragraphs)

async def main():
    """Main function with enhanced error handling and logging."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='PDF Chat System')
    parser.add_argument('pdf_path', nargs='?', help='Path to the PDF file')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Setup logging based on debug flag
    setup_logging(args.debug)
    
    if args.debug:
        logger.info("Starting application in debug mode")
    
    try:
        if not args.pdf_path:
            print("Please provide a PDF file path")
            return

        pdf_path = args.pdf_path
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            print(f"File not found: {pdf_path}")
            return

        # Initialize components
        resource_manager = ResourceManager()
        pdf_processor = PDFProcessor(resource_manager)
        chat = MultiModelChat(resource_manager)

        # Process PDF
        print("Processing PDF... This might take a few minutes.")
        docs_generator = pdf_processor.load_pdf_lazy(pdf_path)
        processed_docs = await pdf_processor.process_documents(docs_generator)
        await chat.initialize_vectorstore(processed_docs)
        print("PDF processed successfully!")

        # Main chat loop
        while True:
            try:
                question = input("\nAsk a question (or type 'exit' to quit): ").strip()
                
                if question.lower() == 'exit':
                    break
                    
                if not question:
                    continue

                print("Thinking...")
                response_dict = await chat.get_response(question)
                
                print("\nCombined Answer:", response_dict["combined_response"])
                
                # Optional: Show individual model responses
                show_details = input("\nShow individual model responses? (y/n): ").lower() == 'y'
                if show_details:
                    print("\nIndividual Model Responses:")
                    for model, response in response_dict["individual_responses"].items():
                        if response:
                            print(f"\n{model.upper()}:")
                            print(response)

            except Exception as e:
                logger.exception("Error in chat loop")
                print(f"An error occurred: {str(e)}")
                print("Please try again.")

    except Exception as e:
        logger.exception("Fatal error in main")
        print(f"A fatal error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.exception("Unhandled exception")
        print(f"Unhandled error: {str(e)}")
from typing import List, Dict, Generator, Optional
import asyncio
import logging
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass
import fitz  # PyMuPDF
from langchain_core.documents import Document

from ..utils.resources import ResourceManager
from ..utils.indexer import DocumentIndexer, Chapter, Paragraph

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """Container for processed document information."""
    content: str
    metadata: Dict
    chapter_info: Optional[Dict] = None
    is_header: bool = False
    is_footer: bool = False
    is_toc: bool = False

class DocumentCleaner:
    """Handles document cleaning and normalization."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        # Remove form feeds and other special characters
        text = re.sub(r'\f', '', text)
        
        # Remove repeated whitespace while preserving paragraph breaks
        text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove standalone page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common header/footer patterns
        text = re.sub(r'^\s*Chapter \d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Page \d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    @staticmethod
    def detect_structure(text: str) -> Dict[str, bool]:
        """Detect structural elements in text."""
        return {
            'is_header': bool(re.match(r'^\s*(?:chapter|section|\d+\.)', text.lower())),
            'is_footer': bool(re.search(r'\d+\s*$', text) and len(text.split()) < 5),
            'is_toc': bool(re.search(r'(?:contents|table of contents)', text.lower()))
                     or bool(re.search(r'\.\s*\.\s*\.\s*\d+$', text))
        }

class BatchProcessor:
    """Handles batch processing of documents."""
    
    def __init__(self, thread_count: int, batch_size: int):
        self.thread_count = thread_count
        self.batch_size = batch_size
        self.cleaner = DocumentCleaner()
    
    async def process_in_batches(self, 
                               docs_generator: Generator[Document, None, None]
                               ) -> List[ProcessedDocument]:
        """Process documents in optimized batches."""
        processed_docs = []
        current_batch = []
        total_processed = 0
        
        logger.info("Starting batch processing")
        
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            try:
                async for doc in self._aiter(docs_generator):
                    current_batch.append(doc)
                    total_processed += 1
                    
                    if len(current_batch) >= self.batch_size:
                        logger.debug(f"Processing batch of {len(current_batch)} documents")
                        processed = await self._process_batch(current_batch, executor)
                        processed_docs.extend(processed)
                        logger.debug(f"Batch processed, got {len(processed)} documents")
                        current_batch = []
                
                if current_batch:  # Process remaining documents
                    logger.debug(f"Processing final batch of {len(current_batch)} documents")
                    processed = await self._process_batch(current_batch, executor)
                    processed_docs.extend(processed)
                    logger.debug(f"Final batch processed, got {len(processed)} documents")
            
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                raise
        
        logger.info(f"Batch processing complete. Processed {total_processed} documents, "
                   f"resulting in {len(processed_docs)} processed documents")
        
        return processed_docs
    
    async def _process_batch(self, 
                           batch: List[Document], 
                           executor: ThreadPoolExecutor
                           ) -> List[ProcessedDocument]:
        """Process a batch of documents in parallel."""
        loop = asyncio.get_event_loop()
        sub_batches = np.array_split(batch, min(len(batch), self.thread_count))
        
        tasks = [
            loop.run_in_executor(executor, self._process_sub_batch, sub_batch)
            for sub_batch in sub_batches
        ]
        
        results = await asyncio.gather(*tasks)
        return [doc for batch_result in results for doc in batch_result]
    
    def _process_sub_batch(self, docs: List[Document]) -> List[ProcessedDocument]:
        """Process a sub-batch of documents."""
        processed_docs = []
        for doc in docs:
            try:
                # Clean the text
                cleaned_content = self.cleaner.clean_text(doc.page_content)
                
                # Skip if content is empty after cleaning
                if not cleaned_content.strip():
                    logger.debug("Skipping empty document after cleaning")
                    continue
                    
                # Detect structural elements
                structure_info = self.cleaner.detect_structure(cleaned_content)
                
                # Create processed document
                processed_docs.append(ProcessedDocument(
                    content=cleaned_content,
                    metadata={
                        **doc.metadata,
                        'processed_timestamp': datetime.now().isoformat(),
                        'length': len(cleaned_content),
                        'word_count': len(cleaned_content.split())
                    },
                    **structure_info
                ))
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        
        return processed_docs
    
    @staticmethod
    async def _aiter(generator):
        """Convert a regular generator to an async generator."""
        for item in generator:
            yield item

class PDFProcessor:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.thread_count = max(1, self.resource_manager.resources.cpu_count - 1)
        self.batch_size = self.resource_manager.optimize_batch_size()
        self.batch_processor = BatchProcessor(self.thread_count, self.batch_size)
        self.indexer = DocumentIndexer()
        self.chapters: List[Chapter] = []

    def load_pdf_lazy(self, pdf_path: str) -> Generator[Document, None, None]:
        """Lazily load a PDF document using PyMuPDF and structure by TOC."""
        logger.info(f"Starting PDF load: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"PDF opened successfully. Total pages: {len(doc)}")
            
            # Extract Table of Contents
            toc = doc.get_toc()
            logger.info(f"Table of contents entries: {len(toc) if toc else 0}")
            
            # Create chapter mapping
            chapter_map = {}
            if toc:
                for level, title, page in toc:
                    if level == 1:  # Assuming only top-level entries are chapters
                        chapter_map[page] = title.strip()
                logger.info(f"Found {len(chapter_map)} chapter mappings")
            
            # Process each page
            current_chapter = None
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    
                    if text.strip():
                        # Determine which chapter this page belongs to
                        for start_page in sorted(chapter_map.keys(), reverse=True):
                            if page_num + 1 >= start_page:
                                current_chapter = chapter_map[start_page]
                                break
                        
                        metadata = {
                            "source": pdf_path,
                            "page": page_num + 1,
                            "chapter_title": current_chapter,
                            "word_count": len(text.split())
                        }
                        
                        yield Document(
                            page_content=text,
                            metadata=metadata
                        )
                    else:
                        logger.warning(f"Page {page_num + 1} is empty")
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    continue

            logger.info(f"PDF processing complete. Processed {len(doc)} pages")
            doc.close()
            
        except Exception as e:
            logger.error(f"Error opening PDF: {e}")
            raise

    async def process_documents(self, 
                              docs_generator: Generator[Document, None, None]
                              ) -> List[Document]:
        """Process documents and organize into chapters based on TOC."""
        try:
            logger.info("Starting document processing")
            
            # Initial processing
            logger.info("Beginning batch processing")
            processed_docs = await self.batch_processor.process_in_batches(docs_generator)
            logger.debug(f"Processed docs content: {[doc.content[:50] + '...' if len(doc.content) > 50 else doc.content for doc in processed_docs]}")
            logger.info(f"Batch processing complete. Got {len(processed_docs)} documents")
            
            if not processed_docs:
                logger.error("No documents were processed in batch processing")
                raise ValueError("No documents were processed")
            
            # Filter out TOC and structural elements
            filtered_docs = [
                doc for doc in processed_docs 
                if not (doc.is_toc or doc.is_header or doc.is_footer)
            ]
            logger.debug(f"Filtered docs content: {[doc.content[:50] + '...' if len(doc.content) > 50 else doc.content for doc in filtered_docs]}")
            logger.info(f"After filtering: {len(filtered_docs)} documents remain")
            
            if not filtered_docs:
                logger.error("All documents were filtered out")
                raise ValueError("All documents were filtered out")
            
            # Group documents by chapter
            chapter_docs = {}
            for doc in filtered_docs:
                chapter_title = doc.metadata.get('chapter_title', 'Uncategorized')
                if chapter_title not in chapter_docs:
                    chapter_docs[chapter_title] = []
                chapter_docs[chapter_title].append(doc)
            
            # Create enhanced documents with chapter context
            enhanced_docs = []
            for chapter_title, docs in chapter_docs.items():
                for idx, doc in enumerate(docs):
                    enhanced_docs.append(Document(
                        page_content=doc.content,
                        metadata={
                            'chapter_title': chapter_title,
                            'paragraph_index': idx,
                            'chapter_context': f"Chapter: {chapter_title}",
                            **doc.metadata
                        }
                    ))
            
            logger.debug(f"Enhanced docs content: {[doc.page_content[:50] + '...' if len(doc.page_content) > 50 else doc.page_content for doc in enhanced_docs]}")
            logger.info(f"Enhanced document creation complete. "
                       f"Created {len(enhanced_docs)} documents across "
                       f"{len(chapter_docs)} chapters")
            
            if not enhanced_docs:
                logger.error("No enhanced documents were created")
                raise ValueError("No enhanced documents were created")
                
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            raise

    def get_chapter_statistics(self) -> Dict:
        """Get statistical information about chapters."""
        chapter_stats = {}
        for chapter_title, docs in self.chapter_docs.items():
            total_words = sum(doc.metadata['word_count'] for doc in docs)
            total_paragraphs = len(docs)
            chapter_stats[chapter_title] = {
                'total_paragraphs': total_paragraphs,
                'total_words': total_words
            }
        return chapter_stats

    def get_chapter_content(self, chapter_title: str) -> Optional[List[Document]]:
        """Retrieve specific chapter content by title."""
        return self.chapter_docs.get(chapter_title, None)

    def get_paragraphs_by_chapter(self, 
                                 chapter_title: str, 
                                 start_idx: int = 0, 
                                 end_idx: Optional[int] = None
                                 ) -> List[Document]:
        """Retrieve specific paragraphs from a chapter by title."""
        chapter_content = self.get_chapter_content(chapter_title)
        if not chapter_content:
            return []
            
        end_idx = end_idx or len(chapter_content)
        return chapter_content[start_idx:end_idx]
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import re
import logging
from datetime import datetime
from langchain_core.documents import Document  # Add this import

logger = logging.getLogger(__name__)

@dataclass
class Paragraph:
    """Represents a single paragraph with content and metadata."""
    content: str
    index: int
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                'length': len(self.content),
                'word_count': len(self.content.split()),
                'timestamp': datetime.now().isoformat()
            }
    
    def update_embedding(self, embedding: List[float]):
        """Update paragraph embedding."""
        self.embedding = embedding
        self.metadata['has_embedding'] = True


@dataclass
class Chapter:
    """Represents a book chapter with its paragraphs and metadata."""
    number: str
    title: str
    paragraphs: List[Paragraph] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                'word_count': sum(p.metadata['word_count'] for p in self.paragraphs),
                'total_paragraphs': len(self.paragraphs),
                'timestamp': datetime.now().isoformat()
            }
    
    def add_paragraph(self, paragraph: Paragraph):
        """Add a paragraph to the chapter and update metadata."""
        self.paragraphs.append(paragraph)
        self.metadata['word_count'] = sum(p.metadata['word_count'] for p in self.paragraphs)
        self.metadata['total_paragraphs'] = len(self.paragraphs)

class TextAnalyzer:
    """Analyzes text for various properties and patterns."""
    
    def __init__(self):
        self.sentence_endings = {'.', '!', '?'}
        self.abbreviations = {'mr.', 'mrs.', 'dr.', 'prof.', 'sr.', 'jr.', 'etc.', 'e.g.', 'i.e.'}
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while handling edge cases."""
        # Replace common abbreviations temporarily
        for abbr in self.abbreviations:
            text = text.replace(abbr, abbr.replace('.', '@'))
            
        # Split on sentence endings
        potential_sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in self.sentence_endings and len(current) > 1:
                sentence = ''.join(current).strip()
                if sentence:
                    potential_sentences.append(sentence)
                current = []
        
        if current:  # Add any remaining text
            sentence = ''.join(current).strip()
            if sentence:
                potential_sentences.append(sentence)
        
        # Restore abbreviations and clean up
        sentences = []
        for sentence in potential_sentences:
            for abbr in self.abbreviations:
                sentence = sentence.replace(abbr.replace('.', '@'), abbr)
            sentences.append(sentence)
            
        return sentences
    
    def is_chapter_header(self, text: str) -> bool:
        """Detect if text is likely a chapter header."""
        patterns = [
            r'^(?:chapter|section)\s+\d+(?:\.\d+)*\s*[:\.]\s*.+$',
            r'^\d+(?:\.\d+)*\s+[A-Z].*$',
            r'^(?:CHAPTER|SECTION)\s+\d+(?:\.\d+)*\s*[:\.]\s*.+$'
        ]
        return any(re.match(pattern, text.strip(), re.IGNORECASE) for pattern in patterns)

class DocumentIndexer:
    """Indexes and organizes document content into chapters and paragraphs."""
    
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.chapters: List[Chapter] = []
        self.seen_titles: Set[str] = set()
        
    def index_document(self, documents: List[Document]) -> List[Chapter]:
        """Index documents into chapters and paragraphs."""
        logger.info("Starting document indexing")
        
        current_chapter = None
        current_content = []
        
        for doc in documents:
            content = doc.content.strip()
            
            # Check for new chapter
            chapter_info = self._extract_chapter_info(content[:200])
            
            if chapter_info and self._is_valid_new_chapter(chapter_info):
                # Process previous chapter if it exists
                if current_chapter and current_content:
                    self._process_chapter_content(current_chapter, current_content)
                
                # Start new chapter
                current_chapter = Chapter(
                    number=chapter_info['number'],
                    title=chapter_info['title']
                )
                self.chapters.append(current_chapter)
                current_content = [content]
                
                # Track chapter title
                self.seen_titles.add(chapter_info['title'].lower())
            else:
                current_content.append(content)
        
        # Process the last chapter
        if current_chapter and current_content:
            self._process_chapter_content(current_chapter, current_content)
        
        logger.info(f"Indexed {len(self.chapters)} chapters")
        return self.chapters
    
    def _extract_chapter_info(self, text: str) -> Optional[Dict]:
        """Extract chapter number and title from text."""
        patterns = [
            r'^(?:Chapter|CHAPTER)\s+(\d+(?:\.\d+)?)\s*[.:]\s*(.+)$',
            r'^(\d+(?:\.\d+)?)\s+(.+)$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip())
            if match:
                return {
                    'number': match.group(1),
                    'title': match.group(2).strip()
                }
        return None
    
    def _is_valid_new_chapter(self, chapter_info: Dict) -> bool:
        """Validate if this is a legitimate new chapter."""
        if not chapter_info:
            return False
            
        title_lower = chapter_info['title'].lower()
        
        # Check if title already seen
        if title_lower in self.seen_titles:
            return False
            
        # Check if title is too short or generic
        if len(title_lower.split()) < 2:
            return False
            
        return True
    
    def _process_chapter_content(self, chapter: Chapter, content: List[str]):
        """Process chapter content into paragraphs."""
        combined_content = ' '.join(content)
        paragraphs = self._split_into_paragraphs(combined_content)
        
        for idx, para_text in enumerate(paragraphs):
            if para_text.strip():  # Skip empty paragraphs
                paragraph = Paragraph(
                    content=para_text,
                    index=idx,
                    metadata={
                        'chapter_number': chapter.number,
                        'chapter_title': chapter.title
                    }
                )
                chapter.add_paragraph(paragraph)
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or significant whitespace
        potential_paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter paragraphs
        paragraphs = []
        for para in potential_paragraphs:
            cleaned = re.sub(r'\s+', ' ', para).strip()
            if cleaned and len(cleaned.split()) > 3:  # Skip very short segments
                paragraphs.append(cleaned)
                
        return paragraphs
    
    def get_chapter_by_number(self, chapter_number: str) -> Optional[Chapter]:
        """Retrieve a specific chapter by its number."""
        for chapter in self.chapters:
            if chapter.number == chapter_number:
                return chapter
        return None
    
    def get_paragraphs_by_keyword(self, keyword: str) -> List[Paragraph]:
        """Find paragraphs containing a specific keyword."""
        results = []
        for chapter in self.chapters:
            for paragraph in chapter.paragraphs:
                if keyword.lower() in paragraph.content.lower():
                    results.append(paragraph)
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistical information about the indexed document."""
        return {
            'total_chapters': len(self.chapters),
            'total_paragraphs': sum(len(c.paragraphs) for c in self.chapters),
            'total_words': sum(c.metadata['word_count'] for c in self.chapters),
            'average_paragraphs_per_chapter': sum(len(c.paragraphs) for c in self.chapters) / len(self.chapters) if self.chapters else 0,
            'chapters': [
                {
                    'number': c.number,
                    'title': c.title,
                    'paragraphs': len(c.paragraphs),
                    'words': c.metadata['word_count']
                }
                for c in self.chapters
            ]
        }

    def export_index(self) -> Dict:
        """Export the document index in a structured format."""
        return {
            'metadata': {
                'total_chapters': len(self.chapters),
                'indexed_at': datetime.now().isoformat()
            },
            'chapters': [
                {
                    'number': chapter.number,
                    'title': chapter.title,
                    'metadata': chapter.metadata,
                    'paragraphs': [
                        {
                            'content': p.content,
                            'index': p.index,
                            'metadata': p.metadata
                        }
                        for p in chapter.paragraphs
                    ]
                }
                for chapter in self.chapters
            ]
        }
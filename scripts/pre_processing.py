#!/usr/bin/env python3
"""
HYBRID Preprocessing for THWS MAI RAG
PyMuPDF (fast) + OCR fallback (for screenshots)

Features:
- Tries PyMuPDF first (handles digital PDFs)
- Falls back to OCR for screenshot/scanned PDFs
- Memory efficient with incremental processing
- Character-based chunking (no memory explosion)

Requirements:
    pip install pymupdf pytesseract pdf2image pillow
    
    # Mac:
    brew install tesseract poppler
    
    # Ubuntu (university server):
    sudo apt-get install tesseract-ocr poppler-utils

Usage:
    python pre_processing.py --src source-docs --out preprocessed
"""

# ============================================================
# SUPPRESS WARNINGS FIRST
# ============================================================
import warnings
warnings.filterwarnings('ignore', message='.*FontBBox.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pdfminer')

import argparse
import json
import os
import gc
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Topic classification keywords
TOPIC_KEYWORDS = {
    'course-info': [
        'course', 'curriculum', 'module', 'semester', 'study', 'exam', 'grade', 'credit',
        'lecture', 'laboratory', 'project', 'thesis', 'professor', 'schedule', 'timetable',
        'MAI', 'master', 'artificial intelligence', 'program', 'degree'
    ],
    'admission': [
        'admission', 'application', 'enrollment', 'enrolment', 'entry', 'requirement',
        'bachelor', 'prerequisite', 'document', 'certificate', 'transcript', 'visa',
        'deadline', 'fee', 'tuition', 'scholarship', 'finance'
    ],
    'wurzburg-guide': [
        'würzburg', 'wurzburg', 'city', 'location', 'transport', 'restaurant', 'culture',
        'museum', 'castle', 'residenz', 'main', 'river', 'old town', 'tourist', 'sightseeing'
    ],
    'housing': [
        'housing', 'accommodation', 'dormitory', 'apartment', 'rent', 'room', 'landlord',
        'contract', 'deposit', 'utilities', 'internet', 'furniture', 'studentenwohnheim',
        'wohnheim', 'flat', 'shared'
    ],
    'campus-life': [
        'campus', 'library', 'cafeteria', 'student', 'club', 'organization', 'sport',
        'recreation', 'event', 'portal', 'card', 'ID', 'login', 'account'
    ]
}

class HybridPDFProcessor:
    def __init__(self, chunk_size: int = 750, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.avg_chars_per_token = 4  # Heuristic: ~4 chars per token
        self.stats = {
            'documents': 0,
            'chunks': 0,
            'skipped': 0,
            'pymupdf_success': 0,
            'ocr_fallback': 0,
            'ocr_failed': 0
        }
        
        # Check if OCR is available
        self.ocr_available = self._check_ocr_available()
        
    def _check_ocr_available(self) -> bool:
        """Check if OCR dependencies are installed."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            # Try a simple OCR test
            pytesseract.get_tesseract_version()
            logger.info("✓ OCR available (Tesseract + pdf2image)")
            return True
        except Exception as e:
            logger.warning(f"⚠ OCR not available: {e}")
            logger.warning("  Install with: pip install pytesseract pdf2image")
            logger.warning("  System: brew install tesseract poppler (Mac)")
            logger.warning("         sudo apt-get install tesseract-ocr poppler-utils (Ubuntu)")
            return False
    
    def extract_pdf_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (fast, works for digital PDFs)."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.debug(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def extract_pdf_with_ocr(self, pdf_path: Path) -> str:
        """Extract text using OCR (slow, works for screenshots/scans)."""
        if not self.ocr_available:
            return ""
        
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import Image
            
            logger.info(f"  📷 Running OCR on {pdf_path.name}...")
            
            # Convert PDF pages to images
            images = convert_from_path(str(pdf_path), dpi=300)
            
            text = ""
            for i, image in enumerate(images):
                # OCR each page
                page_text = pytesseract.image_to_string(image, lang='eng+deu')
                text += f"\n{page_text}\n"
                
                if (i + 1) % 5 == 0:
                    logger.info(f"    OCR progress: {i + 1}/{len(images)} pages")
            
            logger.info(f"  ✓ OCR completed: extracted {len(text)} chars from {len(images)} pages")
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """
        Hybrid PDF extraction: PyMuPDF first, OCR fallback.
        """
        # Try PyMuPDF first (fast)
        text = self.extract_pdf_with_pymupdf(pdf_path)
        
        if text and len(text.strip()) > 100:
            # Success with PyMuPDF
            self.stats['pymupdf_success'] += 1
            logger.info(f"  ✓ PyMuPDF extracted {len(text)} chars")
            return text
        
        # PyMuPDF failed or got too little text - try OCR
        logger.info(f"  ⚠ PyMuPDF got only {len(text)} chars, trying OCR...")
        text = self.extract_pdf_with_ocr(pdf_path)
        
        if text and len(text.strip()) > 50:
            self.stats['ocr_fallback'] += 1
            return text
        else:
            self.stats['ocr_failed'] += 1
            return ""
    
    def extract_docx_text(self, docx_path: Path) -> str:
        """Extract text from DOCX."""
        try:
            import docx
            doc = docx.Document(str(docx_path))
            text = '\n'.join([p.text for p in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting {docx_path.name}: {e}")
            return ""
    
    def extract_text_file(self, file_path: Path) -> str:
        """Extract from text files."""
        try:
            return file_path.read_text(encoding='utf-8')
        except:
            try:
                return file_path.read_text(encoding='latin-1')
            except Exception as e:
                logger.error(f"Error reading {file_path.name}: {e}")
                return ""
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = '\n'.join(lines)
        
        # Remove excessive newlines
        while '\n\n\n' in cleaned:
            cleaned = cleaned.replace('\n\n\n', '\n\n')
        
        return cleaned
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using character heuristic."""
        return len(text) // self.avg_chars_per_token
    
    def chunk_text(self, text: str, source_file: str) -> List[Dict]:
        """
        Character-based chunking with sentence boundary detection.
        Memory efficient - no full tokenization.
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Convert token sizes to character estimates
        char_chunk_size = self.chunk_size * self.avg_chars_per_token
        char_overlap = self.overlap * self.avg_chars_per_token
        
        chunks = []
        start = 0
        chunk_index = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + char_chunk_size, text_length)
            
            # Try to end at sentence boundary
            if end < text_length:
                # Look for sentence endings within next 200 chars
                search_end = min(end + 200, text_length)
                text_segment = text[end:search_end]
                
                # Find sentence boundaries
                sentence_end = -1
                for i, char in enumerate(text_segment):
                    if char in '.!?\n':
                        if char == '\n' or (i + 1 < len(text_segment) and text_segment[i + 1] == ' '):
                            sentence_end = i + 1
                            break
                
                if sentence_end != -1:
                    end = min(text_length, end + sentence_end)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': f"{Path(source_file).stem}_chunk_{chunk_index}",
                    'text': chunk_text,
                    'source_file': source_file,
                    'chunk_index': chunk_index,
                    'char_start': start,
                    'char_end': end,
                    'char_count': len(chunk_text),
                    'estimated_tokens': self.estimate_tokens(chunk_text)
                })
                chunk_index += 1
            
            # Move start position with overlap
            if end >= text_length:
                break
            
            start = end - char_overlap
            if chunks and start <= chunks[-1]['char_start']:
                start = end
        
        return chunks
    
    def classify_topic(self, text: str, filename: str) -> str:
        """Classify document topic based on content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        topic_scores = {}
        
        for topic, keywords in TOPIC_KEYWORDS.items():
            score = 0
            # Score based on content
            for keyword in keywords:
                score += text_lower.count(keyword)
            
            # Boost score if filename contains topic indicators
            for keyword in keywords:
                if keyword in filename_lower:
                    score += 5
            
            topic_scores[topic] = score
        
        # Return topic with highest score, or 'general' if no clear match
        if not topic_scores or max(topic_scores.values()) == 0:
            return 'general'
        
        return max(topic_scores, key=topic_scores.get)
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """Process single file with appropriate extraction method."""
        logger.info(f"Processing: {file_path.name}")
        
        suffix = file_path.suffix.lower()
        
        # Extract text based on file type
        if suffix == '.pdf':
            text = self.extract_pdf_text(file_path)
        elif suffix == '.docx':
            text = self.extract_docx_text(file_path)
        elif suffix in ['.txt', '.md']:
            text = self.extract_text_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return []
        
        # Check if extraction succeeded
        if not text or len(text.strip()) == 0:
            logger.warning(f"  ✗ No text extracted, skipping")
            self.stats['skipped'] += 1
            return []
        
        logger.info(f"  ✓ Extracted {len(text)} characters")
        
        # Clean
        text = self.clean_text(text)
        
        # Chunk
        logger.info(f"  Chunking {len(text)} chars (target ~{self.chunk_size * self.avg_chars_per_token} chars/chunk)...")
        chunks = self.chunk_text(text, file_path.name)
        
        if not chunks:
            logger.warning(f"  ✗ No chunks created")
            self.stats['skipped'] += 1
            return []
        
        logger.info(f"  ✓ Created {len(chunks)} chunks")
        
        # Add topic metadata
        for chunk in chunks:
            chunk['topic'] = self.classify_topic(chunk['text'], file_path.name)
        
        self.stats['documents'] += 1
        self.stats['chunks'] += len(chunks)
        
        # Free memory
        del text
        
        return chunks

def main():
    parser = argparse.ArgumentParser(description='Hybrid preprocessing with PyMuPDF + OCR')
    parser.add_argument('--src', default='source-docs', help='Source documents directory')
    parser.add_argument('--out', default='preprocessed', help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=750, help='Target chunk size in tokens')
    parser.add_argument('--overlap', type=int, default=150, help='Overlap between chunks in tokens')
    
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    processor = HybridPDFProcessor(args.chunk_size, args.overlap)
    
    # Find files
    files = []
    for ext in ['.pdf', '.docx', '.txt', '.md']:
        files.extend(src_dir.glob(f'**/*{ext}'))
    
    if not files:
        logger.error(f"No files found in {src_dir}")
        return
    
    logger.info("="*60)
    logger.info("HYBRID PDF PROCESSING")
    logger.info("="*60)
    logger.info(f"Found {len(files)} files")
    logger.info(f"PyMuPDF: Fast extraction for digital PDFs")
    logger.info(f"OCR: Fallback for screenshots/scans")
    logger.info("="*60)
    
    # Process incrementally
    chunks_file = out_dir / 'rag_chunks.jsonl'
    topic_collections = {t: [] for t in list(TOPIC_KEYWORDS.keys()) + ['general']}
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        for idx, file_path in enumerate(sorted(files), 1):
            logger.info(f"\n[{idx}/{len(files)}] {file_path.name}")
            
            # Process file
            chunks = processor.process_file(file_path)
            
            # Write immediately
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                topic_collections[chunk['topic']].append(chunk)
            
            # Clear memory
            del chunks
            
            # GC every 3 files
            if idx % 3 == 0:
                gc.collect()
                logger.info(f"✓ Memory cleanup (processed {idx}/{len(files)} files)")
    
    logger.info(f"\n✅ Written {processor.stats['chunks']} chunks to {chunks_file}")
    
    # Organize by topic
    topic_dir = out_dir / 'chunks_by_topic'
    topic_dir.mkdir(exist_ok=True)
    
    for topic, chunks in topic_collections.items():
        if not chunks:
            continue
        
        topic_subdir = topic_dir / topic
        topic_subdir.mkdir(exist_ok=True)
        
        with open(topic_subdir / f"{topic}_chunks.jsonl", 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Organized chunks by topic in {topic_dir}")
    
    # Save metadata
    metadata = {
        'processing_date': datetime.now().isoformat(),
        'total_files': len(files),
        'total_chunks': processor.stats['chunks'],
        'files_processed': processor.stats['documents'],
        'files_skipped': processor.stats['skipped'],
        'pymupdf_success': processor.stats['pymupdf_success'],
        'ocr_fallback': processor.stats['ocr_fallback'],
        'ocr_failed': processor.stats['ocr_failed'],
        'chunk_size_tokens': args.chunk_size,
        'overlap_tokens': args.overlap,
        'files': [str(f.name) for f in files]
    }
    
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Files found: {len(files)}")
    logger.info(f"Files processed: {processor.stats['documents']}")
    logger.info(f"Files skipped: {processor.stats['skipped']}")
    logger.info(f"")
    logger.info(f"Extraction methods:")
    logger.info(f"  PyMuPDF success: {processor.stats['pymupdf_success']}")
    logger.info(f"  OCR fallback: {processor.stats['ocr_fallback']}")
    logger.info(f"  OCR failed: {processor.stats['ocr_failed']}")
    logger.info(f"")
    logger.info(f"Total chunks: {processor.stats['chunks']}")
    if processor.stats['documents'] > 0:
        logger.info(f"Avg chunks per file: {processor.stats['chunks'] / processor.stats['documents']:.1f}")
    logger.info(f"")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Main chunks file: {chunks_file}")
    logger.info("="*60)

if __name__ == '__main__':
    main()
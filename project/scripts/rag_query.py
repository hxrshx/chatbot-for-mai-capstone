#!/usr/bin/env python3
"""
Complete Production-Ready RAG System for THWS MAI

This single file includes ALL THREE critical fixes:
1. ✅ Cross-topic search (no topic confinement)
2. ✅ Ambiguity handling (accommodation = housing, not exam)
3. ✅ Query expansion (housing = accommodation = apartment)

Just use this one file - it has everything you need!

Usage:
    python complete_production_rag.py --query "accommodation"
    python complete_production_rag.py --interactive
"""

import argparse
import json
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteProductionRAG:
    """
    Production-ready RAG system with all fixes:
    - Cross-topic search (always searches all topics)
    - Ambiguity handling (disambiguates queries like "accommodation")
    - Query expansion (handles synonyms: housing = accommodation = apartment)
    """
    
    def __init__(self, embeddings_dir: str = "preprocessed/embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        self.model = None
        self.index = None
        self.chunks = None
        self.model_info = None
        
        # ============================================================
        # FIX 3: Query Expansion - Synonym Dictionary
        # ============================================================
        self.synonyms = {
            # Housing (English + German)
            'housing': ['accommodation', 'apartment', 'flat', 'room', 'dormitory', 'dorm',
                       'wohnung', 'zimmer', 'unterkunft', 'wohnheim'],
            'accommodation': ['housing', 'apartment', 'flat', 'room', 'dormitory',
                            'wohnung', 'unterkunft'],
            'apartment': ['flat', 'housing', 'accommodation', 'unit', 'wohnung'],
            'dormitory': ['dorm', 'residence hall', 'student housing', 'wohnheim'],
            'rent': ['rental', 'lease', 'miete', 'mieten'],
            
            # Academic
            'course': ['class', 'module', 'subject', 'lecture', 'kurs', 'vorlesung'],
            'courses': ['classes', 'modules', 'subjects', 'lectures', 'kurse'],
            'exam': ['test', 'examination', 'prüfung', 'klausur'],
            'grade': ['mark', 'score', 'note', 'bewertung'],
            'study': ['studieren', 'learn', 'lernen'],
            'professor': ['lecturer', 'instructor', 'dozent'],
            
            # Admission
            'apply': ['application', 'enroll', 'register', 'bewerbung', 'anmeldung'],
            'application': ['apply', 'enrollment', 'registration', 'bewerbung'],
            'admission': ['entry', 'enrollment', 'zulassung'],
            'enroll': ['register', 'sign up', 'anmelden', 'einschreiben'],
            'visa': ['visum', 'permit'],
            
            # Campus
            'library': ['bibliothek', 'study center'],
            'cafeteria': ['canteen', 'mensa'],
            'student': ['scholar', 'studierende'],
            'campus': ['university grounds', 'hochschule'],
        }
        
        # Common query patterns
        self.query_patterns = {
            'where to live': ['housing', 'accommodation', 'apartment'],
            'place to stay': ['housing', 'accommodation'],
            'find accommodation': ['housing', 'apartment', 'dormitory'],
            'how to apply': ['application', 'admission', 'enrollment'],
            'what courses': ['courses', 'classes', 'curriculum'],
        }
        
        # ============================================================
        # FIX 2: Ambiguity Handling - Ambiguous Terms
        # ============================================================
        self.ambiguous_terms = {
            'accommodation': {
                'housing': ['student housing', 'apartments', 'dormitory', 'rent'],
                'exam': ['reasonable accommodation', 'disability', 'testing']
            },
            'program': {
                'academic': ['degree program', 'mai program', 'curriculum'],
                'software': ['software program', 'application']
            },
            'application': {
                'admission': ['university application', 'apply', 'enrollment'],
                'software': ['app', 'software', 'tool']
            }
        }
        
        # Context keywords for disambiguation
        self.context_keywords = {
            'housing': ['housing', 'apartment', 'flat', 'room', 'rent', 'dormitory',
                       'wohnung', 'zimmer', 'live', 'stay'],
            'academic': ['course', 'study', 'exam', 'grade', 'lecture', 'semester',
                        'kurs', 'studium', 'prüfung'],
            'admission': ['admission', 'application', 'enroll', 'apply', 'visa',
                         'bewerbung', 'zulassung'],
            'exam_accommodation': ['reasonable', 'disability', 'special needs',
                                  'testing', 'adjustment']
        }
    
    def load_system(self):
        """Load RAG system components."""
        logger.info("Loading complete production RAG system...")
        
        # Load model info
        model_info_file = self.embeddings_dir / "model_info.json"
        if model_info_file.exists():
            with open(model_info_file, 'r') as f:
                self.model_info = json.load(f)
        
        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.model_info['model_name'] if self.model_info else 'intfloat/multilingual-e5-small'
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Load FAISS index
        try:
            import faiss
            index_file = self.embeddings_dir / "embeddings.index"
            self.index = faiss.read_index(str(index_file))
            logger.info(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
        
        # Load chunks
        try:
            metadata_file = self.embeddings_dir / "chunk_metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            logger.info(f"✅ Loaded chunks: {len(self.chunks)}")
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            raise
        
        logger.info("="*60)
        logger.info("✅ COMPLETE RAG SYSTEM READY")
        logger.info("   Fix 1: Cross-topic search enabled")
        logger.info("   Fix 2: Ambiguity handling enabled")
        logger.info("   Fix 3: Query expansion enabled")
        logger.info("="*60)
    
    # ============================================================
    # FIX 3: Query Expansion Methods
    # ============================================================
    
    def expand_query(self, query: str, max_expansions: int = 3) -> str:
        """Expand query with synonyms."""
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Check common patterns first
        for pattern, expansions in self.query_patterns.items():
            if pattern in query_lower:
                expansion_str = ' '.join(expansions[:max_expansions])
                expanded = f"{query} {expansion_str}"
                logger.info(f"📝 Pattern match: '{query}' → '{expanded}'")
                return expanded
        
        # Collect synonyms
        expansion_terms = set()
        original_terms = set(words)
        
        for word in words:
            if len(word) <= 2:  # Skip short words
                continue
            if word in self.synonyms:
                synonyms = self.synonyms[word][:max_expansions]
                expansion_terms.update(synonyms)
        
        # Remove duplicates
        expansion_terms = expansion_terms - original_terms
        
        if expansion_terms:
            expanded = f"{query} {' '.join(list(expansion_terms)[:max_expansions * 2])}"
            logger.info(f"📝 Synonym expansion: '{query}' → '{expanded}'")
            return expanded
        
        return query
    
    # ============================================================
    # FIX 2: Ambiguity Handling Methods
    # ============================================================
    
    def detect_ambiguous_query(self, query: str) -> Optional[Dict]:
        """Detect if query contains ambiguous terms."""
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Single-word queries are often ambiguous
        if len(words) == 1 and words[0] in self.ambiguous_terms:
            return {
                'is_ambiguous': True,
                'term': words[0],
                'possible_meanings': list(self.ambiguous_terms[words[0]].keys())
            }
        
        # Check for ambiguous terms without context
        for term in self.ambiguous_terms.keys():
            if term in query_lower:
                has_context = False
                for meaning, keywords in self.ambiguous_terms[term].items():
                    if any(kw in query_lower for kw in keywords):
                        has_context = True
                        break
                
                if not has_context:
                    return {
                        'is_ambiguous': True,
                        'term': term,
                        'possible_meanings': list(self.ambiguous_terms[term].keys())
                    }
        
        return None
    
    def disambiguate_query(self, query: str) -> Tuple[str, str]:
        """Disambiguate query based on context."""
        query_lower = query.lower()
        
        # Score each context category
        context_scores = {}
        for context_type, keywords in self.context_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                context_scores[context_type] = score
        
        primary_intent = max(context_scores, key=context_scores.get) if context_scores else 'general'
        
        # Enhance query if ambiguous
        ambiguity = self.detect_ambiguous_query(query)
        enhanced_query = query
        
        if ambiguity:
            term = ambiguity['term']
            intent_to_meaning = {
                'housing': 'housing',
                'academic': 'academic',
                'admission': 'admission',
                'exam_accommodation': 'exam'
            }
            
            meaning = intent_to_meaning.get(primary_intent)
            if meaning and term in self.ambiguous_terms:
                expansion_terms = self.ambiguous_terms[term].get(meaning, [])
                if expansion_terms:
                    enhanced_query = f"{query} {' '.join(expansion_terms[:3])}"
                    logger.info(f"🎯 Disambiguated: '{query}' → '{enhanced_query}' (intent: {primary_intent})")
        
        return enhanced_query, primary_intent
    
    def apply_keyword_boosting(self, results: List[Dict], query: str, intent: str) -> List[Dict]:
        """Apply keyword-based score boosting."""
        boost_keywords = []
        penalty_keywords = []
        
        if intent == 'housing':
            boost_keywords = ['housing', 'apartment', 'flat', 'room', 'rent', 'dormitory',
                            'wohnung', 'zimmer', 'live', 'stay']
            penalty_keywords = ['exam', 'test', 'grade', 'reasonable accommodation',
                              'disability', 'special needs']
        elif intent == 'exam_accommodation':
            boost_keywords = ['reasonable', 'disability', 'special needs', 'testing', 'exam']
            penalty_keywords = ['housing', 'apartment', 'rent']
        
        for result in results:
            text_lower = result['text'].lower()
            original_score = result['similarity_score']
            
            boost_count = sum(1 for kw in boost_keywords if kw in text_lower)
            penalty_count = sum(1 for kw in penalty_keywords if kw in text_lower)
            
            adjustment = (boost_count * 0.05) - (penalty_count * 0.10)
            adjusted = original_score + adjustment
            
            result['original_score'] = original_score
            result['similarity_score'] = max(0.0, min(1.0, adjusted))
            result['score_adjustment'] = adjustment
        
        # Re-sort
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
    
    # ============================================================
    # FIX 1: Cross-Topic Search (Always Search ALL Topics)
    # ============================================================
    
    def search_all_topics(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search across ALL topics (never confined to one).
        This is Fix #1 - no more topic silos!
        """
        if not self.model or not self.index:
            raise RuntimeError("System not loaded. Call load_system() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search entire corpus (not topic-specific)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 4)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                chunk['initial_rank'] = i + 1
                results.append(chunk)
        
        logger.info(f"🌍 Searched ALL topics: {len(results)} candidates found")
        
        # Log topic distribution
        topic_dist = {}
        for r in results[:top_k]:
            topic = r.get('topic', 'unknown')
            topic_dist[topic] = topic_dist.get(topic, 0) + 1
        logger.info(f"📊 Results from topics: {topic_dist}")
        
        return {
            'candidates': results,
            'topic_distribution': topic_dist
        }
    
    # ============================================================
    # Complete Pipeline: All Three Fixes Together
    # ============================================================
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG pipeline with all three fixes.
        
        Pipeline:
        1. Expand query with synonyms (Fix 3)
        2. Disambiguate if ambiguous (Fix 2)
        3. Search ALL topics (Fix 1)
        4. Apply keyword boosting (Fix 2)
        5. Return top-k results
        """
        if not self.model:
            self.load_system()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Processing query: '{query_text}'")
        logger.info(f"{'='*60}")
        
        # FIX 3: Expand query with synonyms
        expanded_query = self.expand_query(query_text)
        
        # FIX 2: Disambiguate if ambiguous
        ambiguity_info = self.detect_ambiguous_query(expanded_query)
        disambiguated_query, detected_intent = self.disambiguate_query(expanded_query)
        
        # FIX 1: Search ALL topics (never confined)
        search_results = self.search_all_topics(disambiguated_query, top_k)
        candidates = search_results['candidates']
        
        # FIX 2: Apply keyword boosting
        if detected_intent != 'general':
            candidates = self.apply_keyword_boosting(candidates, query_text, detected_intent)
        
        # Take top-k
        final_results = candidates[:top_k]
        
        # Update ranks
        for i, r in enumerate(final_results):
            r['final_rank'] = i + 1
        
        response = {
            'original_query': query_text,
            'expanded_query': expanded_query if expanded_query != query_text else None,
            'disambiguated_query': disambiguated_query if disambiguated_query != expanded_query else None,
            'ambiguity_detected': ambiguity_info,
            'detected_intent': detected_intent,
            'searched_all_topics': True,  # Always true now!
            'topic_distribution': search_results['topic_distribution'],
            'num_results': len(final_results),
            'results': final_results,
            'formatted_results': self.format_results(final_results),  # Add this!
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Returned {len(final_results)} results")
        logger.info(f"{'='*60}\n")
        
        return response
    
    def build_context(self, results: List[Dict], max_tokens: int = 3000) -> str:
        """Build token-aware context for LLM."""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            
            context_parts = []
            current_tokens = 0
            
            for result in results:
                text = result['text']
                source_info = f"\n[Source: {result['source_file']}, Topic: {result['topic']}]\n"
                full_chunk = source_info + text
                
                chunk_tokens = len(encoding.encode(full_chunk))
                
                if current_tokens + chunk_tokens > max_tokens:
                    remaining = max_tokens - current_tokens
                    if remaining > 100:
                        partial = encoding.decode(encoding.encode(text)[:remaining - 50])
                        context_parts.append(source_info + partial + "\n[...]")
                    break
                
                context_parts.append(full_chunk)
                current_tokens += chunk_tokens
            
            return "\n\n".join(context_parts)
            
        except ImportError:
            logger.warning("tiktoken not available, using character limit")
            context = []
            for r in results:
                context.append(f"[Source: {r['source_file']}]\n{r['text']}")
            return "\n\n".join(context)[:max_tokens * 4]
    
    def format_results(self, results: List[Dict]) -> str:
        """Format results for display."""
        if not results:
            return "No results found."
        
        formatted = []
        for r in results:
            rank = f"Rank {r['final_rank']}"
            if 'initial_rank' in r and r['initial_rank'] != r['final_rank']:
                rank += f" (was #{r['initial_rank']})"
            
            score = f"Score: {r['similarity_score']:.3f}"
            if 'score_adjustment' in r and r['score_adjustment'] != 0:
                score += f" (adjusted {r['score_adjustment']:+.3f})"
            
            header = f"📄 {r['source_file']} | {score} | {rank}"
            topic = f"🏷️  Topic: {r.get('topic', 'unknown')}"
            text = r['text'][:300] + "..." if len(r['text']) > 300 else r['text']
            
            formatted.append(f"{header}\n{topic}\n\n{text}\n" + "="*60)
        
        return "\n\n".join(formatted)


def main():
    parser = argparse.ArgumentParser(
        description='Complete Production RAG System (All Fixes Included)'
    )
    parser.add_argument('--query', help='Query text')
    parser.add_argument('--embeddings-dir', default='preprocessed/embeddings')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--interactive', action='store_true')
    
    args = parser.parse_args()
    
    rag = CompleteProductionRAG(embeddings_dir=args.embeddings_dir)
    
    if args.interactive:
        print("\n" + "="*70)
        print("🎓 COMPLETE PRODUCTION RAG SYSTEM")
        print("="*70)
        print("\n✨ All fixes included:")
        print("   ✅ Cross-topic search (no information silos)")
        print("   ✅ Ambiguity handling (correct interpretation)")
        print("   ✅ Query expansion (synonym consistency)")
        print("\nType 'quit' to exit\n")
        
        while True:
            try:
                query = input("💬 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif not query:
                    continue
                
                response = rag.query(query, args.top_k)
                
                # Show processing info
                if response.get('expanded_query'):
                    print(f"\n📝 Expanded: {response['expanded_query']}")
                if response.get('disambiguated_query'):
                    print(f"🎯 Disambiguated: {response['disambiguated_query']}")
                if response.get('ambiguity_detected'):
                    amb = response['ambiguity_detected']
                    print(f"⚠️  Ambiguous: '{amb['term']}' - {amb['possible_meanings']}")
                
                print(f"🌍 Searched: ALL topics")
                print(f"📊 Results from: {response['topic_distribution']}")
                print(f"🎯 Intent: {response['detected_intent']}")
                print(f"\n📖 Results:\n")
                print(response['formatted_results'])
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    elif args.query:
        response = rag.query(args.query, args.top_k)
        
        print(f"\n🔍 Query: '{response['original_query']}'")
        
        if response.get('expanded_query'):
            print(f"📝 Expanded: {response['expanded_query']}")
        if response.get('disambiguated_query'):
            print(f"🎯 Disambiguated: {response['disambiguated_query']}")
        if response.get('ambiguity_detected'):
            amb = response['ambiguity_detected']
            print(f"⚠️  Ambiguous: '{amb['term']}' - meanings: {amb['possible_meanings']}")
        
        print(f"🌍 Searched: ALL topics")
        print(f"📊 Results from: {response['topic_distribution']}")
        print(f"🎯 Intent: {response['detected_intent']}")
        print(f"\n📖 Results:\n")
        print(rag.format_results(response['results']))
    
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("This RAG system includes ALL THREE fixes:")
        print("  1. Cross-topic search (always searches all topics)")
        print("  2. Ambiguity handling (accommodation = housing, not exam)")
        print("  3. Query expansion (housing = accommodation = apartment)")
        print("="*70)


if __name__ == '__main__':
    main()
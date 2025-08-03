"""
RAG (Retrieval-Augmented Generation) pipeline for the LLM-Powered Q&A System.
"""

import time
from typing import List, Dict, Any, Optional
from ..utils.logging import get_logger, log_performance
from ..utils.config import get_config
from .llm_model import LLMModel
from .prompt_manager import PromptManager
from ..embedding.embedding_pipeline import EmbeddingPipeline


class RAGPipeline:
    """
    Main RAG pipeline that coordinates retrieval and generation.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = get_config("models")
        
        self.config = config
        self.logger = get_logger("rag_pipeline")
        
        # Initialize components
        self.embedding_pipeline = EmbeddingPipeline(config)
        self.llm_model = LLMModel(config=config)
        self.prompt_manager = PromptManager(config)
        
        # Pipeline settings
        self.rag_config = config.get("rag", {})
        self.max_context_length = self.rag_config.get("max_context_length", 4000)
        self.context_window = self.rag_config.get("context_window", 2000)
    
    def answer_question(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: The question to answer
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the answer and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info("Processing question", question=question)
            
            # Step 1: Retrieve relevant documents
            search_results = self.embedding_pipeline.search(question, k)
            
            if not search_results:
                self.logger.warning("No relevant documents found", question=question)
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "retrieved_documents": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Step 2: Format context from search results
            context = self.prompt_manager.format_context(search_results)
            
            # Step 3: Generate answer
            answer = self.llm_model.generate_with_context(context, question)
            
            # Step 4: Prepare response
            duration = time.time() - start_time
            
            # Extract source information
            sources = []
            for result in search_results:
                metadata = result.get("metadata", {})
                source_info = {
                    "source": metadata.get("source", "unknown"),
                    "title": metadata.get("title", "Unknown"),
                    "url": metadata.get("url", ""),
                    "similarity": result.get("similarity", 0.0)
                }
                sources.append(source_info)
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(r.get("similarity", 0.0) for r in search_results) / len(search_results)
            
            response = {
                "answer": answer,
                "sources": sources,
                "confidence": avg_similarity,
                "retrieved_documents": len(search_results),
                "processing_time": duration,
                "question": question,
                "context_length": len(context)
            }
            
            self.logger.info(
                "Question answered successfully",
                question=question,
                answer_length=len(answer),
                retrieved_documents=len(search_results),
                confidence=avg_similarity,
                processing_time=duration
            )
            
            log_performance(
                "rag_question_answering",
                duration,
                question_length=len(question),
                answer_length=len(answer),
                retrieved_documents=len(search_results)
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.logger.error(
                "Failed to answer question",
                question=question,
                error=str(e),
                processing_time=duration,
                exc_info=True
            )
            
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "retrieved_documents": 0,
                "processing_time": duration,
                "error": str(e)
            }
    
    def batch_answer_questions(self, questions: List[str], k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Answer multiple questions using the RAG pipeline.
        
        Args:
            questions: List of questions to answer
            k: Number of documents to retrieve per question
            
        Returns:
            List of answer dictionaries
        """
        start_time = time.time()
        
        try:
            self.logger.info("Processing batch questions", question_count=len(questions))
            
            answers = []
            
            for i, question in enumerate(questions):
                try:
                    answer = self.answer_question(question, k)
                    answers.append(answer)
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to answer question in batch",
                        question_index=i,
                        question=question,
                        error=str(e)
                    )
                    
                    # Add error response
                    answers.append({
                        "answer": "I encountered an error while processing this question.",
                        "sources": [],
                        "confidence": 0.0,
                        "retrieved_documents": 0,
                        "processing_time": 0.0,
                        "error": str(e)
                    })
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Batch question processing completed",
                question_count=len(questions),
                successful_answers=len([a for a in answers if "error" not in a]),
                processing_time=duration
            )
            
            log_performance(
                "batch_rag_question_answering",
                duration,
                question_count=len(questions)
            )
            
            return answers
            
        except Exception as e:
            self.logger.error(
                "Batch question processing failed",
                question_count=len(questions),
                error=str(e),
                exc_info=True
            )
            raise
    
    def summarize_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Summarize text using the LLM.
        
        Args:
            text: Text to summarize
            max_length: Maximum length for summary
            
        Returns:
            Generated summary
        """
        try:
            prompt = self.prompt_manager.create_summary_prompt(text, max_length)
            summary = self.llm_model.generate(prompt)
            
            self.logger.info(
                "Text summarized successfully",
                original_length=len(text),
                summary_length=len(summary)
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(
                "Text summarization failed",
                text_length=len(text),
                error=str(e),
                exc_info=True
            )
            raise
    
    def analyze_text(self, text: str, analysis_type: str = "general") -> str:
        """
        Analyze text using the LLM.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (general, technical, key_points)
            
        Returns:
            Generated analysis
        """
        try:
            prompt = self.prompt_manager.create_analysis_prompt(text, analysis_type)
            analysis = self.llm_model.generate(prompt)
            
            self.logger.info(
                "Text analysis completed",
                text_length=len(text),
                analysis_type=analysis_type,
                analysis_length=len(analysis)
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(
                "Text analysis failed",
                text_length=len(text),
                analysis_type=analysis_type,
                error=str(e),
                exc_info=True
            )
            raise
    
    def compare_texts(self, text1: str, text2: str) -> str:
        """
        Compare two texts using the LLM.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Generated comparison
        """
        try:
            prompt = self.prompt_manager.create_comparison_prompt(text1, text2)
            comparison = self.llm_model.generate(prompt)
            
            self.logger.info(
                "Text comparison completed",
                text1_length=len(text1),
                text2_length=len(text2),
                comparison_length=len(comparison)
            )
            
            return comparison
            
        except Exception as e:
            self.logger.error(
                "Text comparison failed",
                text1_length=len(text1),
                text2_length=len(text2),
                error=str(e),
                exc_info=True
            )
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG pipeline.
        
        Returns:
            Dictionary containing pipeline information
        """
        return {
            "embedding_pipeline": self.embedding_pipeline.get_pipeline_info(),
            "llm_model": self.llm_model.get_model_info(),
            "prompt_manager": self.prompt_manager.get_prompt_info(),
            "rag_config": {
                "max_context_length": self.max_context_length,
                "context_window": self.context_window
            }
        }
    
    def load_index(self, index_path: str):
        """
        Load an existing index.
        
        Args:
            index_path: Path to the index file
        """
        self.embedding_pipeline.load_index(index_path)
    
    def clear_index(self):
        """Clear the current index."""
        self.embedding_pipeline.clear_index() 
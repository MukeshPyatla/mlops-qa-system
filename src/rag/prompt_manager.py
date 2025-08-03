"""
Prompt manager for the LLM-Powered Q&A System.
"""

from typing import Dict, Any, List, Optional
from ..utils.logging import get_logger
from ..utils.config import get_config


class PromptManager:
    """
    Manages prompt templates and formatting for the RAG pipeline.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the prompt manager.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = get_config("models")
        
        self.config = config.get("rag", {})
        self.logger = get_logger("prompt_manager")
        
        # Load prompt templates
        self.system_prompt = self.config.get("system_prompt", "")
        self.user_prompt_template = self.config.get("user_prompt_template", "")
        
        # Context settings
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.context_window = self.config.get("context_window", 2000)
    
    def create_qa_prompt(self, question: str, context: str, 
                        include_system_prompt: bool = True) -> str:
        """
        Create a Q&A prompt with context.
        
        Args:
            question: The question to answer
            context: Retrieved context information
            include_system_prompt: Whether to include system prompt
            
        Returns:
            Formatted prompt string
        """
        # Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        # Format the prompt
        if include_system_prompt and self.system_prompt:
            prompt = f"{self.system_prompt}\n\n"
        else:
            prompt = ""
        
        # Use template if available, otherwise use default format
        if self.user_prompt_template:
            prompt += self.user_prompt_template.format(
                context=context,
                question=question
            )
        else:
            prompt += f"""Context: {context}

Question: {question}

Answer:"""
        
        return prompt
    
    def create_summary_prompt(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Create a prompt for text summarization.
        
        Args:
            text: Text to summarize
            max_length: Maximum length for summary
            
        Returns:
            Formatted prompt string
        """
        max_length = max_length or self.context_window
        
        # Truncate text if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = f"""Please provide a concise summary of the following text:

{text}

Summary:"""
        
        return prompt
    
    def create_analysis_prompt(self, text: str, analysis_type: str = "general") -> str:
        """
        Create a prompt for text analysis.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (general, technical, key_points)
            
        Returns:
            Formatted prompt string
        """
        analysis_instructions = {
            "general": "Please analyze the following text and provide insights:",
            "technical": "Please provide a technical analysis of the following text:",
            "key_points": "Please extract the key points from the following text:"
        }
        
        instruction = analysis_instructions.get(analysis_type, analysis_instructions["general"])
        
        # Truncate text if too long
        if len(text) > self.max_context_length:
            text = text[:self.max_context_length] + "..."
        
        prompt = f"""{instruction}

{text}

Analysis:"""
        
        return prompt
    
    def create_comparison_prompt(self, text1: str, text2: str) -> str:
        """
        Create a prompt for comparing two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Formatted prompt string
        """
        # Truncate texts if too long
        max_length = self.max_context_length // 2
        
        if len(text1) > max_length:
            text1 = text1[:max_length] + "..."
        if len(text2) > max_length:
            text2 = text2[:max_length] + "..."
        
        prompt = f"""Please compare the following two texts and highlight the key differences and similarities:

Text 1:
{text1}

Text 2:
{text2}

Comparison:"""
        
        return prompt
    
    def format_context(self, search_results: List[Dict[str, Any]], 
                      max_context_length: Optional[int] = None) -> str:
        """
        Format search results into context for the LLM.
        
        Args:
            search_results: List of search results with text and metadata
            max_context_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        max_length = max_context_length or self.max_context_length
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results):
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            similarity = result.get("similarity", 0.0)
            
            # Format this result
            source = metadata.get("source", "unknown")
            title = metadata.get("title", f"Document {i+1}")
            
            formatted_result = f"[Source: {source}, Title: {title}, Similarity: {similarity:.2f}]\n{text}\n"
            
            # Check if adding this would exceed the limit
            if current_length + len(formatted_result) > max_length:
                break
            
            context_parts.append(formatted_result)
            current_length += len(formatted_result)
        
        return "\n".join(context_parts)
    
    def create_follow_up_prompt(self, original_question: str, answer: str, 
                               follow_up_question: str) -> str:
        """
        Create a prompt for follow-up questions.
        
        Args:
            original_question: The original question
            answer: The answer to the original question
            follow_up_question: The follow-up question
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Based on the previous question and answer:

Original Question: {original_question}
Answer: {answer}

Follow-up Question: {follow_up_question}

Please provide an answer to the follow-up question, building on the previous context:"""
        
        return prompt
    
    def get_prompt_info(self) -> Dict[str, Any]:
        """
        Get information about the prompt manager.
        
        Returns:
            Dictionary containing prompt manager information
        """
        return {
            "system_prompt_length": len(self.system_prompt),
            "user_prompt_template_length": len(self.user_prompt_template),
            "max_context_length": self.max_context_length,
            "context_window": self.context_window,
            "has_system_prompt": bool(self.system_prompt),
            "has_user_template": bool(self.user_prompt_template)
        } 
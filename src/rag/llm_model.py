"""
LLM model for the LLM-Powered Q&A System.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional
import time

from ..utils.logging import get_logger, log_performance
from ..utils.config import get_config


class LLMModel:
    """
    Language model for text generation in the RAG pipeline.
    """
    
    def __init__(self, model_name: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize the LLM model.
        
        Args:
            model_name: Name of the model to use
            config: Configuration dictionary
        """
        if config is None:
            config = get_config("models")
        
        self.config = config.get("llm", {})
        self.model_name = model_name or self.config.get("model", "mistralai/Mistral-7B-Instruct-v0.2")
        self.max_tokens = self.config.get("max_tokens", 512)
        self.temperature = self.config.get("temperature", 0.7)
        self.top_p = self.config.get("top_p", 0.9)
        self.repetition_penalty = self.config.get("repetition_penalty", 1.1)
        
        # Model loading settings
        self.device_map = self.config.get("device_map", "auto")
        self.torch_dtype = self.config.get("torch_dtype", "auto")
        self.load_in_8bit = self.config.get("load_in_8bit", False)
        self.load_in_4bit = self.config.get("load_in_4bit", False)
        
        self.logger = get_logger("llm_model")
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model."""
        try:
            self.logger.info("Loading LLM model", model=self.model_name)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device_map
            )
            
            self.logger.info(
                "LLM model loaded successfully",
                model=self.model_name,
                device_map=self.device_map
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load LLM model",
                model=self.model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        try:
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            self.logger.info(
                "Generating text",
                prompt_length=len(prompt),
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Generate text
            outputs = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Text generation completed",
                input_length=len(prompt),
                output_length=len(generated_text),
                duration_seconds=duration
            )
            
            log_performance(
                "text_generation",
                duration,
                input_length=len(prompt),
                output_length=len(generated_text),
                max_tokens=max_tokens
            )
            
            return generated_text
            
        except Exception as e:
            self.logger.error(
                "Text generation failed",
                prompt_length=len(prompt),
                error=str(e),
                exc_info=True
            )
            raise
    
    def generate_with_context(self, context: str, question: str, 
                            max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None) -> str:
        """
        Generate an answer based on context and question.
        
        Args:
            context: Context information
            question: Question to answer
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        # Create prompt with context and question
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        
        return self.generate(prompt, max_tokens, temperature)
    
    def batch_generate(self, prompts: List[str], 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated texts
        """
        start_time = time.time()
        
        try:
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            self.logger.info(
                "Batch text generation",
                prompt_count=len(prompts),
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_texts = []
            
            for i, prompt in enumerate(prompts):
                try:
                    generated_text = self.generate(prompt, max_tokens, temperature)
                    generated_texts.append(generated_text)
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to generate text for prompt",
                        prompt_index=i,
                        error=str(e)
                    )
                    generated_texts.append("")
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Batch text generation completed",
                prompt_count=len(prompts),
                successful_count=len([t for t in generated_texts if t]),
                duration_seconds=duration
            )
            
            log_performance(
                "batch_text_generation",
                duration,
                prompt_count=len(prompts)
            )
            
            return generated_texts
            
        except Exception as e:
            self.logger.error(
                "Batch text generation failed",
                prompt_count=len(prompts),
                error=str(e),
                exc_info=True
            )
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "device_map": self.device_map,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit
        } 
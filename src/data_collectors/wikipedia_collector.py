"""
Wikipedia collector for the LLM-Powered Q&A System.
"""

import asyncio
import wikipediaapi
from typing import List, Dict, Any
import wikipedia

from .base_collector import BaseDataCollector
from ..utils.logging import get_logger


class WikipediaCollector(BaseDataCollector):
    """
    Collector for Wikipedia articles.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.sources = config.get("sources", {}).get("wikipedia", [])
        self.wiki_api = None
        
        # Initialize Wikipedia API
        for source in self.sources:
            if "language" in source:
                lang = source["language"]
                self.wiki_api = wikipediaapi.Wikipedia(lang)
                break
        
        if not self.wiki_api:
            self.wiki_api = wikipediaapi.Wikipedia('en')
    
    async def collect(self) -> List[Dict[str, Any]]:
        """
        Collect Wikipedia articles from configured topics.
        
        Returns:
            List of collected Wikipedia articles
        """
        collected_data = []
        
        for source in self.sources:
            try:
                source_data = await self._collect_from_source(source)
                collected_data.extend(source_data)
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect from Wikipedia source",
                    source=source.get("name", "unknown"),
                    error=str(e),
                    exc_info=True
                )
        
        return collected_data
    
    async def _collect_from_source(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect Wikipedia articles from a single source configuration.
        
        Args:
            source: Source configuration
            
        Returns:
            List of collected articles
        """
        topics = source.get("topics", [])
        max_pages = source.get("max_pages_per_topic", 5)
        language = source.get("language", "en")
        
        collected_articles = []
        
        for topic in topics:
            try:
                articles = await self._collect_topic_articles(topic, max_pages, language)
                collected_articles.extend(articles)
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect topic articles",
                    topic=topic,
                    error=str(e)
                )
        
        return collected_articles
    
    async def _collect_topic_articles(self, topic: str, max_pages: int, language: str) -> List[Dict[str, Any]]:
        """
        Collect articles related to a specific topic.
        
        Args:
            topic: Topic to search for
            max_pages: Maximum number of pages to collect
            language: Language code
            
        Returns:
            List of collected articles
        """
        self.logger.info("Collecting Wikipedia articles", topic=topic, max_pages=max_pages)
        
        articles = []
        
        try:
            # Search for the topic
            search_results = wikipedia.search(topic, results=max_pages)
            
            for page_title in search_results:
                try:
                    # Get the page
                    page = self.wiki_api.page(page_title)
                    
                    if page.exists():
                        article_data = self._process_wikipedia_page(page, topic)
                        if article_data:
                            articles.append(article_data)
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to process Wikipedia page",
                        page_title=page_title,
                        error=str(e)
                    )
                    continue
            
            # Also get the main topic page if it exists
            main_page = self.wiki_api.page(topic)
            if main_page.exists():
                main_article = self._process_wikipedia_page(main_page, topic)
                if main_article:
                    articles.append(main_article)
            
        except Exception as e:
            self.logger.error(
                "Failed to search Wikipedia",
                topic=topic,
                error=str(e)
            )
        
        self.logger.info(
            "Collected Wikipedia articles",
            topic=topic,
            article_count=len(articles)
        )
        
        return articles
    
    def _process_wikipedia_page(self, page, topic: str) -> Dict[str, Any]:
        """
        Process a Wikipedia page and extract relevant information.
        
        Args:
            page: Wikipedia page object
            topic: Original search topic
            
        Returns:
            Processed article data
        """
        # Extract text content
        content = page.text
        
        # Skip if content is too short
        if len(content) < 100:
            return None
        
        # Extract summary (first few paragraphs)
        summary = page.summary
        
        # Extract categories
        categories = list(page.categories.keys()) if hasattr(page, 'categories') else []
        
        # Extract links
        links = list(page.links.keys()) if hasattr(page, 'links') else []
        
        # Create data item
        data_item = self.create_data_item(
            content=content,
            url=page.fullurl,
            title=page.title,
            source="wikipedia",
            category="encyclopedia",
            topic=topic,
            summary=summary,
            categories=categories[:10],  # Limit to first 10 categories
            links=links[:20],  # Limit to first 20 links
            language=page.language
        )
        
        return data_item
    
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the Wikipedia sources.
        
        Returns:
            Dictionary containing source information
        """
        all_topics = []
        for source in self.sources:
            all_topics.extend(source.get("topics", []))
        
        return {
            "type": "wikipedia",
            "topics": all_topics,
            "total_topics": len(all_topics),
            "language": self.sources[0].get("language", "en") if self.sources else "en"
        } 
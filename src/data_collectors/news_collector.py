"""
News collector for the LLM-Powered Q&A System.
"""

import asyncio
import feedparser
import aiohttp
from typing import List, Dict, Any
from datetime import datetime
from bs4 import BeautifulSoup

from .base_collector import BaseDataCollector
from ..utils.logging import get_logger


class NewsCollector(BaseDataCollector):
    """
    Collector for news articles from RSS feeds.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.sources = config.get("sources", {}).get("news", [])
        self.session = None
    
    async def collect(self) -> List[Dict[str, Any]]:
        """
        Collect news articles from configured RSS feeds.
        
        Returns:
            List of collected news articles
        """
        collected_data = []
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            for source in self.sources:
                try:
                    source_data = await self._collect_from_source(source)
                    collected_data.extend(source_data)
                    
                except Exception as e:
                    self.logger.error(
                        "Failed to collect from news source",
                        source=source.get("name", "unknown"),
                        error=str(e),
                        exc_info=True
                    )
        
        return collected_data
    
    async def _collect_from_source(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect news articles from a single RSS feed.
        
        Args:
            source: Source configuration
            
        Returns:
            List of collected articles
        """
        rss_feed = source["rss_feed"]
        source_name = source.get("name", rss_feed)
        max_articles = source.get("max_articles", 50)
        
        self.logger.info("Collecting news articles", source=source_name, feed=rss_feed)
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(rss_feed)
            
            if feed.bozo:
                self.logger.warning("RSS feed has parsing errors", source=source_name)
            
            articles = []
            
            # Process each entry
            for entry in feed.entries[:max_articles]:
                try:
                    article_data = await self._process_news_entry(entry, source)
                    if article_data:
                        articles.append(article_data)
                        
                except Exception as e:
                    self.logger.warning(
                        "Failed to process news entry",
                        source=source_name,
                        title=entry.get("title", "unknown"),
                        error=str(e)
                    )
                    continue
            
            self.logger.info(
                "Collected news articles",
                source=source_name,
                article_count=len(articles)
            )
            
            return articles
            
        except Exception as e:
            self.logger.error(
                "Failed to parse RSS feed",
                source=source_name,
                feed=rss_feed,
                error=str(e)
            )
            return []
    
    async def _process_news_entry(self, entry, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single news entry from RSS feed.
        
        Args:
            entry: RSS feed entry
            source: Source configuration
            
        Returns:
            Processed article data
        """
        # Extract basic information
        title = entry.get("title", "")
        link = entry.get("link", "")
        published = entry.get("published", "")
        summary = entry.get("summary", "")
        
        # Try to get full content
        content = await self._fetch_full_content(link, summary)
        
        # Parse published date
        published_date = None
        if published:
            try:
                # Try to parse the date
                published_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
            except:
                try:
                    published_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                except:
                    published_date = datetime.now()
        else:
            published_date = datetime.now()
        
        # Extract author
        author = entry.get("author", "")
        if not author and hasattr(entry, 'authors') and entry.authors:
            author = entry.authors[0].get("name", "")
        
        # Extract tags/categories
        tags = []
        if hasattr(entry, 'tags'):
            tags = [tag.term for tag in entry.tags]
        elif hasattr(entry, 'category'):
            tags = [entry.category]
        
        # Create data item
        data_item = self.create_data_item(
            content=content,
            url=link,
            title=title,
            source=source.get("name", "news"),
            category="news",
            author=author,
            published_date=published_date.isoformat() if published_date else None,
            tags=tags,
            summary=summary
        )
        
        return data_item
    
    async def _fetch_full_content(self, url: str, summary: str) -> str:
        """
        Fetch full content from article URL.
        
        Args:
            url: Article URL
            summary: RSS summary as fallback
            
        Returns:
            Full article content
        """
        if not url:
            return summary
        
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse HTML to extract main content
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Try to find main content area
                    main_content = None
                    
                    # Common selectors for article content
                    selectors = [
                        'article',
                        '.article-content',
                        '.post-content',
                        '.entry-content',
                        '.content',
                        'main',
                        '#content'
                    ]
                    
                    for selector in selectors:
                        main_content = soup.select_one(selector)
                        if main_content:
                            break
                    
                    if main_content:
                        # Extract text from main content
                        text_content = main_content.get_text(separator=' ', strip=True)
                        
                        # If we got meaningful content, return it
                        if len(text_content) > len(summary):
                            return text_content
                    
                    # Fallback to summary
                    return summary
                    
        except Exception as e:
            self.logger.warning(
                "Failed to fetch full content",
                url=url,
                error=str(e)
            )
        
        # Return summary as fallback
        return summary
    
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the news sources.
        
        Returns:
            Dictionary containing source information
        """
        source_names = [source.get("name", source.get("rss_feed", "unknown")) for source in self.sources]
        
        return {
            "type": "news",
            "sources": source_names,
            "total_sources": len(self.sources),
            "update_frequency": "hourly"  # News is typically updated frequently
        } 
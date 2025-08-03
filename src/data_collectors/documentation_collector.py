"""
Documentation collector for the LLM-Powered Q&A System.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import markdown
from urllib.parse import urljoin, urlparse

from .base_collector import BaseDataCollector
from ..utils.logging import get_logger


class DocumentationCollector(BaseDataCollector):
    """
    Collector for documentation websites.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.sources = config.get("sources", {}).get("documentation", [])
        self.session = None
    
    async def collect(self) -> List[Dict[str, Any]]:
        """
        Collect documentation from configured sources.
        
        Returns:
            List of collected documentation items
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
                        "Failed to collect from source",
                        source=source.get("name", "unknown"),
                        error=str(e),
                        exc_info=True
                    )
        
        return collected_data
    
    async def _collect_from_source(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect documentation from a single source.
        
        Args:
            source: Source configuration
            
        Returns:
            List of collected items from this source
        """
        url = source["url"]
        source_name = source.get("name", url)
        selector = source.get("selector", "body")
        content_type = source.get("type", "html")
        
        self.logger.info("Collecting from source", source=source_name, url=url)
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(
                        "Failed to fetch source",
                        source=source_name,
                        status=response.status
                    )
                    return []
                
                content = await response.text()
                
                if content_type == "markdown":
                    return await self._process_markdown(content, source)
                else:
                    return await self._process_html(content, source)
                    
        except Exception as e:
            self.logger.error(
                "Error fetching source",
                source=source_name,
                url=url,
                error=str(e)
            )
            return []
    
    async def _process_html(self, content: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process HTML content from a documentation source.
        
        Args:
            content: HTML content
            source: Source configuration
            
        Returns:
            List of processed items
        """
        soup = BeautifulSoup(content, 'html.parser')
        selector = source.get("selector", "body")
        
        # Find the main content area
        main_content = soup.select_one(selector)
        if not main_content:
            self.logger.warning("No content found with selector", selector=selector)
            return []
        
        # Extract text content
        text_content = main_content.get_text(separator=' ', strip=True)
        
        # Extract title
        title_elem = soup.find('title')
        title = title_elem.get_text() if title_elem else source.get("name", "")
        
        # Extract links for additional pages
        links = []
        for link in main_content.find_all('a', href=True):
            href = link.get('href')
            if href and not href.startswith('#') and not href.startswith('javascript:'):
                full_url = urljoin(source["url"], href)
                link_text = link.get_text(strip=True)
                if link_text:
                    links.append({
                        "url": full_url,
                        "text": link_text
                    })
        
        # Create data item
        data_item = self.create_data_item(
            content=text_content,
            url=source["url"],
            title=title,
            source=source.get("name", "documentation"),
            category="documentation",
            links=links[:10]  # Limit to first 10 links
        )
        
        return [data_item]
    
    async def _process_markdown(self, content: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process Markdown content from a documentation source.
        
        Args:
            content: Markdown content
            source: Source configuration
            
        Returns:
            List of processed items
        """
        # Convert markdown to HTML first
        html_content = markdown.markdown(content)
        
        # Then process as HTML
        return await self._process_html(html_content, source)
    
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the documentation sources.
        
        Returns:
            Dictionary containing source information
        """
        return {
            "type": "documentation",
            "sources": [source.get("name", source.get("url", "unknown")) for source in self.sources],
            "total_sources": len(self.sources)
        } 
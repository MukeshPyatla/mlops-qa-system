"""
GitHub collector for the LLM-Powered Q&A System.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any
from datetime import datetime

from .base_collector import BaseDataCollector
from ..utils.logging import get_logger


class GitHubCollector(BaseDataCollector):
    """
    Collector for GitHub repository information.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.sources = config.get("sources", {}).get("github", [])
        self.session = None
        self.github_token = None  # Could be set from environment
        
    async def collect(self) -> List[Dict[str, Any]]:
        """
        Collect GitHub repository information from configured sources.
        
        Returns:
            List of collected repository data
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
                        "Failed to collect from GitHub source",
                        source=source.get("name", "unknown"),
                        error=str(e),
                        exc_info=True
                    )
        
        return collected_data
    
    async def _collect_from_source(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect repository information from a single source configuration.
        
        Args:
            source: Source configuration
            
        Returns:
            List of collected repository data
        """
        search_terms = source.get("search_terms", [])
        language = source.get("language", "Python")
        sort_by = source.get("sort", "stars")
        max_repos = source.get("max_repos", 20)
        
        collected_repos = []
        
        for term in search_terms:
            try:
                repos = await self._search_repositories(term, language, sort_by, max_repos)
                collected_repos.extend(repos)
                
            except Exception as e:
                self.logger.error(
                    "Failed to search repositories",
                    term=term,
                    error=str(e)
                )
        
        return collected_repos
    
    async def _search_repositories(self, search_term: str, language: str, 
                                  sort_by: str, max_repos: int) -> List[Dict[str, Any]]:
        """
        Search for repositories on GitHub.
        
        Args:
            search_term: Search term
            language: Programming language filter
            sort_by: Sort criteria (stars, forks, updated)
            max_repos: Maximum number of repositories to return
            
        Returns:
            List of repository data
        """
        self.logger.info(
            "Searching GitHub repositories",
            search_term=search_term,
            language=language,
            sort_by=sort_by
        )
        
        # Build search query
        query = f"{search_term} language:{language}"
        
        # GitHub API endpoint
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": sort_by,
            "order": "desc",
            "per_page": min(max_repos, 100)  # GitHub API limit
        }
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "LLM-QA-System/1.0"
        }
        
        # Add authentication if token is available
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    repositories = data.get("items", [])
                    
                    # Process each repository
                    processed_repos = []
                    for repo in repositories[:max_repos]:
                        try:
                            repo_data = await self._process_repository(repo, search_term)
                            if repo_data:
                                processed_repos.append(repo_data)
                        except Exception as e:
                            self.logger.warning(
                                "Failed to process repository",
                                repo_name=repo.get("full_name", "unknown"),
                                error=str(e)
                            )
                            continue
                    
                    self.logger.info(
                        "Collected GitHub repositories",
                        search_term=search_term,
                        repo_count=len(processed_repos)
                    )
                    
                    return processed_repos
                    
                else:
                    self.logger.error(
                        "GitHub API request failed",
                        status=response.status,
                        search_term=search_term
                    )
                    return []
                    
        except Exception as e:
            self.logger.error(
                "Failed to search GitHub",
                search_term=search_term,
                error=str(e)
            )
            return []
    
    async def _process_repository(self, repo: Dict[str, Any], search_term: str) -> Dict[str, Any]:
        """
        Process a single GitHub repository.
        
        Args:
            repo: Repository data from GitHub API
            search_term: Original search term
            
        Returns:
            Processed repository data
        """
        # Extract basic information
        name = repo.get("name", "")
        full_name = repo.get("full_name", "")
        description = repo.get("description", "")
        language = repo.get("language", "")
        url = repo.get("html_url", "")
        api_url = repo.get("url", "")
        
        # Get detailed repository information
        detailed_info = await self._get_repository_details(api_url)
        
        # Combine basic and detailed information
        content = f"""
Repository: {full_name}
Description: {description}
Language: {language}
Stars: {repo.get('stargazers_count', 0)}
Forks: {repo.get('forks_count', 0)}
Watchers: {repo.get('watchers_count', 0)}
Created: {repo.get('created_at', '')}
Updated: {repo.get('updated_at', '')}
Homepage: {repo.get('homepage', '')}
License: {repo.get('license', {}).get('name', 'Unknown') if repo.get('license') else 'Unknown'}

{detailed_info.get('readme_content', '')}

Topics: {', '.join(detailed_info.get('topics', []))}
"""
        
        # Create data item
        data_item = self.create_data_item(
            content=content,
            url=url,
            title=full_name,
            source="github",
            category="repository",
            search_term=search_term,
            language=language,
            stars=repo.get('stargazers_count', 0),
            forks=repo.get('forks_count', 0),
            topics=detailed_info.get('topics', []),
            readme_content=detailed_info.get('readme_content', ''),
            description=description
        )
        
        return data_item
    
    async def _get_repository_details(self, api_url: str) -> Dict[str, Any]:
        """
        Get detailed information about a repository.
        
        Args:
            api_url: GitHub API URL for the repository
            
        Returns:
            Dictionary containing detailed repository information
        """
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "LLM-QA-System/1.0"
        }
        
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        try:
            # Get topics
            topics_url = f"{api_url}/topics"
            async with self.session.get(topics_url, headers=headers) as response:
                topics = []
                if response.status == 200:
                    topics_data = await response.json()
                    topics = topics_data.get("names", [])
            
            # Get README content
            readme_url = f"{api_url}/readme"
            readme_content = ""
            async with self.session.get(readme_url, headers=headers) as response:
                if response.status == 200:
                    readme_data = await response.json()
                    readme_content = readme_data.get("content", "")
                    # Decode base64 content if needed
                    if readme_content:
                        import base64
                        readme_content = base64.b64decode(readme_content).decode('utf-8')
            
            return {
                "topics": topics,
                "readme_content": readme_content
            }
            
        except Exception as e:
            self.logger.warning(
                "Failed to get repository details",
                api_url=api_url,
                error=str(e)
            )
            return {
                "topics": [],
                "readme_content": ""
            }
    
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the GitHub sources.
        
        Returns:
            Dictionary containing source information
        """
        all_search_terms = []
        for source in self.sources:
            all_search_terms.extend(source.get("search_terms", []))
        
        return {
            "type": "github",
            "search_terms": all_search_terms,
            "total_search_terms": len(all_search_terms),
            "language": self.sources[0].get("language", "Python") if self.sources else "Python"
        } 
"""
Web Search Integration — DuckDuckGo.
Provides real-time information retrieval for up-to-date responses.
No API key required.
"""

import asyncio
import time
from typing import Optional

from loguru import logger


class WebSearchEngine:
    """
    Web search engine using DuckDuckGo for real-time information.
    Fetches search results and formats them as context for LLM prompts.
    """

    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """Verify DuckDuckGo search is available."""
        try:
            from duckduckgo_search import DDGS
            self._initialized = True
            logger.success("Web search engine (DuckDuckGo) initialized")
        except ImportError:
            logger.error("duckduckgo-search not installed — run: pip install duckduckgo-search")
            self._initialized = False

    async def search(
        self,
        query: str,
        max_results: int = 3,
        region: str = "wt-wt",
    ) -> list[dict]:
        """
        Perform a web search and return structured results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            region: Region code (wt-wt = no region)

        Returns:
            List of dicts with 'title', 'snippet', 'url'
        """
        if not self._initialized:
            logger.warning("Web search not initialized")
            return []

        start_time = time.time()

        try:
            from duckduckgo_search import DDGS

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._search_sync(query, max_results, region),
            )

            elapsed = int((time.time() - start_time) * 1000)
            logger.debug(f"Web search completed in {elapsed}ms — {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    def _search_sync(self, query: str, max_results: int, region: str) -> list[dict]:
        """Synchronous DuckDuckGo search."""
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results, region=region))

        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            })

        return results

    async def search_news(self, query: str, max_results: int = 3) -> list[dict]:
        """Search specifically for news articles."""
        if not self._initialized:
            return []

        try:
            from duckduckgo_search import DDGS

            loop = asyncio.get_event_loop()

            def _news_sync():
                with DDGS() as ddgs:
                    raw = list(ddgs.news(query, max_results=max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("url", ""),
                        "source": r.get("source", ""),
                        "date": r.get("date", ""),
                    }
                    for r in raw
                ]

            return await loop.run_in_executor(None, _news_sync)

        except Exception as e:
            logger.error(f"News search error: {e}")
            return []

    def format_results_for_prompt(self, results: list[dict]) -> str:
        """
        Format search results into a context string for LLM prompt injection.

        Args:
            results: List of search result dicts

        Returns:
            Formatted string ready for prompt injection
        """
        if not results:
            return "No web search results were found."

        formatted = "=== WEB SEARCH RESULTS ===\n"
        for i, r in enumerate(results, 1):
            formatted += f"\n[{i}] {r['title']}\n"
            formatted += f"    {r['snippet']}\n"
            formatted += f"    Source: {r['url']}\n"

        formatted += "\n=== END SEARCH RESULTS ===\n"
        return formatted

    @property
    def is_ready(self) -> bool:
        return self._initialized

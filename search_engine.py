from typing import List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

class SearchItem(BaseModel):
    """Represents a single search result item"""

    title: str = Field(description="The title of the search result")
    url: str = Field(description="The URL of the search result")
    description: Optional[str] = Field(
        default=None, description="A description or snippet of the search result"
    )

    def __str__(self) -> str:
        """String representation of a search result item."""
        return f"{self.title} - {self.url}"


class WebSearchEngine(BaseModel):
    """Base class for web search engines."""

    model_config = {"arbitrary_types_allowed": True}

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Perform a web search and return a list of search items.

        Args:
            query (str): The search query to submit to the search engine.
            num_results (int, optional): The number of search results to return. Default is 10.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            List[SearchItem]: A list of SearchItem objects matching the search query.
        """
        raise NotImplementedError

ABSTRACT_MAX_LENGTH = 300

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/49.0.2623.108 Chrome/49.0.2623.108 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; pt-BR) AppleWebKit/533.3 (KHTML, like Gecko) QtWeb Internet Browser/3.7 http://www.QtWeb.net",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.2 (KHTML, like Gecko) ChromePlus/4.0.222.3 Chrome/4.0.222.3 Safari/532.2",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.4pre) Gecko/20070404 K-Ninja/2.1.3",
    "Mozilla/5.0 (Future Star Technologies Corp.; Star-Blade OS; x86_64; U; en-US) iNet Browser 4.7",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.13) Gecko/20080414 Firefox/2.0.0.13 Pogo/2.0.0.13.6866",
]

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": USER_AGENTS[0],
    "Referer": "https://www.bing.com/",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

BING_HOST_URL = "https://www.bing.com"
BING_SEARCH_URL = "https://www.bing.com/search?q="

class BingSearchEngine(WebSearchEngine):
    session: Optional[requests.Session] = None

    def __init__(self, **data):
        """Initialize the BingSearch tool with a requests session."""
        super().__init__(**data)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _search_sync(self, query: str, num_results: int = 10) -> List[SearchItem]:
        """
        Synchronous Bing search implementation to retrieve search results.

        Args:
            query (str): The search query to submit to Bing.
            num_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            List[SearchItem]: A list of search items with title, URL, and description.
        """
        if not query:
            return []

        list_result = []
        first = 1
        next_url = BING_SEARCH_URL + query

        while len(list_result) < num_results:
            data, next_url = self._parse_html(
                next_url, rank_start=len(list_result), first=first
            )
            if data:
                list_result.extend(data)
            if not next_url:
                break
            first += 10

        return list_result[:num_results]

    def _parse_html(
        self, url: str, rank_start: int = 0, first: int = 1
    ) -> Tuple[List[SearchItem], str]:
        """
        Parse Bing search result HTML to extract search results and the next page URL.

        Returns:
            tuple: (List of SearchItem objects, next page URL or None)
        """
        try:
            res = self.session.get(url=url)
            res.encoding = "utf-8"
            root = BeautifulSoup(res.text, "lxml")

            list_data = []
            ol_results = root.find("ol", id="b_results")
            if not ol_results:
                return [], None

            for li in ol_results.find_all("li", class_="b_algo"):
                title = ""
                url = ""
                abstract = ""
                try:
                    h2 = li.find("h2")
                    if h2:
                        title = h2.text.strip()
                        url = h2.a["href"].strip()

                    p = li.find("p")
                    if p:
                        abstract = p.text.strip()

                    # if ABSTRACT_MAX_LENGTH and len(abstract) > ABSTRACT_MAX_LENGTH:
                    #     abstract = abstract[:ABSTRACT_MAX_LENGTH]

                    rank_start += 1

                    # Create a SearchItem object
                    list_data.append(
                        SearchItem(
                            title=title or f"Bing Result {rank_start}",
                            url=url,
                            description=abstract,
                        )
                    )
                except Exception:
                    continue

            next_btn = root.find("a", title="Next page")
            if not next_btn:
                return list_data, None

            next_url = BING_HOST_URL + next_btn["href"]
            return list_data, next_url
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return [], None

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Bing search engine.

        Returns results formatted according to SearchItem model.
        """
        return self._search_sync(query, num_results=num_results)
    
    def run(self, query: str, num_results: int = 10) -> str:
        results = self.perform_search(query,num_results)
        if not results:
            return "No results found."
        summary = "\n".join([f"{i+1}. {r.title}\n{r.url}\n{r.description}\n" for i, r in enumerate(results)])
        return f"Top {len(results)} Bing results for '{query}':\n" + summary



from duckduckgo_search import DDGS

class DuckDuckGoSearchEngine(WebSearchEngine):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        DuckDuckGo search engine.

        Returns results formatted according to SearchItem model.
        """
        raw_results = DDGS().text(query, max_results=num_results)

        results = []
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                # If it's just a URL
                results.append(
                    SearchItem(
                        title=f"DuckDuckGo Result {i + 1}", url=item, description=None
                    )
                )
            elif isinstance(item, dict):
                # Extract data from the dictionary
                results.append(
                    SearchItem(
                        title=item.get("title", f"DuckDuckGo Result {i + 1}"),
                        url=item.get("href", ""),
                        description=item.get("body", None),
                    )
                )
            else:
                # Try to extract attributes directly
                try:
                    results.append(
                        SearchItem(
                            title=getattr(item, "title", f"DuckDuckGo Result {i + 1}"),
                            url=getattr(item, "href", ""),
                            description=getattr(item, "body", None),
                        )
                    )
                except Exception:
                    # Fallback
                    results.append(
                        SearchItem(
                            title=f"DuckDuckGo Result {i + 1}",
                            url=str(item),
                            description=None,
                        )
                    )

        return results

    def run(self, query: str, num_results: int = 10) -> str:
        results = self.perform_search(query,num_results)
        if not results:
            return "No results found."
        summary = "\n".join([f"{i+1}. {r.title}\n{r.url}\n{r.description}\n" for i, r in enumerate(results)])
        return f"Top {len(results)} Bing results for '{query}':\n" + summary

my_search=DuckDuckGoSearchEngine()
print(my_search.run("Tsinghua university"))
from typing import List, Optional, Tuple, Dict, Any
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



from ddgs import DDGS

class DDGSSearchEngine(WebSearchEngine):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        DuckDuckGo search engine.

        Returns results formatted according to SearchItem model.
        """
        raw_results = list(DDGS().text(query, max_results=num_results))
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
        # if not results:
        #     return "No results found."
        # results = [{"title":r.title,"url":r.url,"description":r.description}for r in results]
        # return results
        # if not results:
        #     return "No results found."
        summary = "\n".join([f"{i+1}. {r.title}\n{r.url}\n{r.description}\n" for i, r in enumerate(results)])
        with open('./search_result', 'w', encoding='utf-8') as f:
            f.write(f"Top {len(results)} DDGS results for '{query}':\n" + summary)
        return f"Top {len(results)} DDGS results for '{query}':\n" + summary




import asyncio
import requests
import time


class GoogleSearchEngine(WebSearchEngine):
    
    async def _search_with_google(self, query: str, max_links_per_query: int=100) -> List[str]:
        """Performs a web search using the Google Custom Search API and returns a list of URLs."""

        api_key = "AIzaSyDbFzATgCgaY4koEh5d9sDGPj-ihh5XQeM" 
        cse_id = "b2dc2984b77a24d20"
        print(f"api_key: {api_key}, cse_id: {cse_id}")

        if not api_key or not cse_id:
            print("Google Custom Search API key or CSE ID is not configured. Please set them in your config file.")
            return []

        # Google Custom Search API endpoint
        base_url = "https://www.googleapis.com/customsearch/v1"

        # Truncate the query to avoid errors
        truncated_query = query[:100]
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": truncated_query,
            "num": min(max_links_per_query, 10)  # Google Custom Search allows max 10 results per request
        }

        retry_count = 0
        links = []
        while retry_count < 3:
            try:
                response = requests.get(
                    base_url,
                    params=params,
                    timeout=30,
                )
                if response.status_code != 200:
                    print(
                        f"Google Custom Search API returned status code {response.status_code}. "
                        f"Response: {response.text}"
                    )
                    time.sleep(0.5)
                    retry_count += 1
                else:
                    data = response.json()
                    results = data.get("items", [])
                    links = [item.get("link") for item in results if item.get("link")]
                    print(f"Google search found {len(links)} links.")
                    break
            except requests.exceptions.RequestException as e:
                print(f"Google Custom Search API request failed: {e}")
                time.sleep(0.5)
                retry_count += 1
        return links


    async def _get_section_search_content(self, search_query: str, num_results: int) -> List[Dict[str, Any]]:
        """Fetches content from the web based on a search query."""
        print(f"Searching for: '{search_query}'")

        links = await self._search_with_google(search_query, max_links_per_query=num_results)
        if not links:
            print(f"No links found for '{search_query}'.")
            return []

        # Extract content from links
        collected_content = []
        # extraction_tasks = [extract_main_content(link) for link in links]
        extraction_tasks = links
        extracted_contents = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        for link, content in zip(links, extracted_contents):
            if isinstance(content, str) and content:
                collected_content.append({"url": link, "content": content})
                print(f"Successfully extracted content from {link}.")
            elif isinstance(content, dict) and content:
                collected_content.append({"url": link, "content": content.get("markdown_content")})
                print(f"Successfully extracted structured content from {link}.")
            elif isinstance(content, list) and content:
                for item in content:
                    collected_content.append({"url": link, "content": item.get("markdown_content")})
                print(f"Successfully extracted multiple sections from {link}.")
            else:
                print(f"Failed to extract valid content from {link}.")

        return collected_content

    async def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Google search engine.

        Returns results formatted according to SearchItem model.
        """
        # raw_results = search(query, num_results=num_results, advanced=True, api_key=config.google_search.api_key, cse_id=config.google_search.cse_id)
        raw_results = await self._get_section_search_content(query, num_results)
        results = []
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                # If it's just a URL
                results.append(
                    {"title": f"Google Result {i+1}", "url": item, "description": ""}
                )
            else:
                results.append(
                    SearchItem(
                        title=item['content'], url=item['url'], description=item['content']
                    )
                )

        print(results)

if __name__ == '__main__':
    asyncio.run(GoogleSearchEngine().perform_search("What is the capital of France?", num_results=5))

# my_search=DDGSSearchEngine()
# with open('./search_result','w',encoding='utf-8') as f:
#     f.write(str(my_search.run("MMA featherweight 14 significant strikes 83 attempted 16.87% swordsman nickname")))
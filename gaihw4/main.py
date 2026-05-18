import argparse
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import json
import os
import random
import time
import re
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count

START = 8151
END = 9600
OUTPUT_DIR = "output"


def format_mmdd(date_text):
    if not date_text:
        return None
    parts = date_text.strip().split("/")
    if len(parts) != 2:
        return None
    month, day = parts
    try:
        return f"{int(month):02d}{int(day):02d}"
    except ValueError:
        return None


def crawl():
    base_url = "https://www.ptt.cc"
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "articles.jsonl")
    popular_path = os.path.join(output_dir, "popular_articles.jsonl")

    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(popular_path):
        os.remove(popular_path)

    session = requests.Session()
    session.cookies.update({"over18": "1"})
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        }
    )
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    with open(output_path, "a", encoding="utf-8") as fout, open(popular_path, "a", encoding="utf-8") as fpop:
        for page_index in tqdm(range(START, END + 1), desc="Crawling pages"):
            page_url = f"{base_url}/bbs/Stock/index{page_index}.html"
            try:
                resp = session.get(page_url, timeout=10)
                resp.raise_for_status()
            except Exception as exc:
                print(f"warning: could not fetch page {page_index}: {exc}")
                time.sleep(1 + random.random())
                continue
            time.sleep(0.2 + random.random() * 0.2)

            soup = BeautifulSoup(resp.text, "html.parser")
            for entry in soup.find_all("div", class_="r-ent"):
                title_div = entry.find("div", class_="title")
                if not title_div:
                    continue

                a_tag = title_div.find("a")
                if not a_tag:
                    continue

                title = a_tag.text.strip()
                if not title or "[公告]" in title or "Fw:[公告]" in title:
                    continue

                url = base_url + a_tag["href"]
                date_div = entry.find("div", class_="date")
                date_mmdd = format_mmdd(date_div.text if date_div else "")
                if not date_mmdd:
                    continue
                if page_index == START and date_mmdd == "1231":
                    continue
                if page_index == END and date_mmdd == "0101":
                    continue

                nrec_div = entry.find("div", class_="nrec")
                is_popular = False
                if nrec_div:
                    nrec_content = nrec_div.get_text(strip=True)
                    if "爆" in nrec_content:
                        is_popular = True

                article = {"date": date_mmdd, "title": title, "url": url}
                fout.write(json.dumps(article, ensure_ascii=False) + "\n")
                
                if is_popular:
                    fpop.write(json.dumps(article, ensure_ascii=False) + "\n")


def fetch_and_parse_article(url):
    """Fetch and parse an article URL to extract push/boo statistics."""
    session = requests.Session()
    session.cookies.update({"over18": "1"})
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        }
    )
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        return {"push_counts": {}, "boo_counts": {}}
    
    soup = BeautifulSoup(resp.text, "html.parser")
    push_counts = defaultdict(int)
    boo_counts = defaultdict(int)
    
    # Find all push divs
    for push_div in soup.find_all("div", class_="push"):
        # Extract the tag (推/嘘/→) - find span with "push-tag" class (may have other classes too)
        tag_span = push_div.find("span", class_=lambda x: x and "push-tag" in x)
        if not tag_span:
            continue
        
        tag_text = tag_span.get_text(strip=True)
        
        # Extract user ID - find span with "push-userid" class (may have other classes too)
        userid_span = push_div.find("span", class_=lambda x: x and "push-userid" in x)
        if not userid_span:
            continue
        
        user_id = userid_span.get_text()  # Keep original formatting with spaces
        
        # Count based on tag (only count 推 and 嘘, not →)
        # Check all possible characters for 嘘/噓
        if "推" in tag_text:
            push_counts[user_id] += 1
        if any(c in tag_text for c in ["嘘", "噓"]):
            boo_counts[user_id] += 1
    
    return {"push_counts": dict(push_counts), "boo_counts": dict(boo_counts)}


def extract_image_urls(url):
    """Extract all image URLs from an article."""
    session = requests.Session()
    session.cookies.update({"over18": "1"})
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    })
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        return []
    
    soup = BeautifulSoup(resp.text, "html.parser")
    image_urls = []
    
    # Valid image extensions
    valid_extensions = (".jpg", ".jpeg", ".png", ".gif")
    
    def is_image_url(test_url):
        """Check if URL is an image URL, handling query params and fragments."""
        if not (test_url.startswith("http://") or test_url.startswith("https://")):
            return False
        # Remove query parameters and fragments
        url_path = test_url.split("?")[0].split("#")[0]
        return any(url_path.lower().endswith(ext) for ext in valid_extensions)
    
    # Check all <a> tags for image links
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if is_image_url(href):
            image_urls.append(href)
    
    # Check all <img> tags for image sources
    for img_tag in soup.find_all("img", src=True):
        src = img_tag["src"]
        if is_image_url(src):
            image_urls.append(src)
    
    # Also extract bare URLs from text (image URLs in plain text like in comments)
    page_text = soup.get_text()
    url_pattern = r'https?://[^\s\)\]]*\.(?:jpg|jpeg|png|gif)'
    for match in re.finditer(url_pattern, page_text, re.IGNORECASE):
        matched_url = match.group(0)
        # Clean up any trailing punctuation
        matched_url = matched_url.rstrip('.,;:)')
        if is_image_url(matched_url):
            image_urls.append(matched_url)
    
    return image_urls


def push(start_date, end_date):
    """Extract push/boo statistics for articles in the date range."""
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    articles_path = os.path.join(output_dir, "articles.jsonl")
    output_path = os.path.join(output_dir, f"push_{start_date}_{end_date}.json")
    
    # Read articles from articles.jsonl
    urls_to_process = []
    with open(articles_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
                article_date = article.get("date", "")
                
                # Check if date is within range (inclusive)
                if start_date <= article_date <= end_date:
                    urls_to_process.append(article.get("url", ""))
            except json.JSONDecodeError:
                continue
    
    # Use multiprocessing to fetch and parse articles
    all_push_counts = defaultdict(int)
    all_boo_counts = defaultdict(int)
    total_push = 0
    total_boo = 0
    
    num_processes = cpu_count()
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(fetch_and_parse_article, urls_to_process),
            total=len(urls_to_process),
            desc="Processing articles"
        ))
    
    # Aggregate results
    for result in results:
        for user_id, count in result["push_counts"].items():
            all_push_counts[user_id] += count
            total_push += count
        
        for user_id, count in result["boo_counts"].items():
            all_boo_counts[user_id] += count
            total_boo += count
    
    # Get top 10 users by push and boo
    top10_push = sorted(all_push_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_boo = sorted(all_boo_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Format output
    output_data = {
        "push": {
            "total": total_push,
            "top10": [{"user_id": user_id, "count": count} for user_id, count in top10_push]
        },
        "boo": {
            "total": total_boo,
            "top10": [{"user_id": user_id, "count": count} for user_id, count in top10_boo]
        }
    }
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Push statistics saved to {output_path}")


def popular(start_date, end_date):
    """Extract image URLs from popular articles in date range."""
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    popular_path = os.path.join(output_dir, "popular_articles.jsonl")
    output_path = os.path.join(output_dir, f"popular_{start_date}_{end_date}.json")
    
    # Read popular articles in date range
    urls_to_process = []
    with open(popular_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
                article_date = article.get("date", "")
                
                # Check if date is within range (inclusive)
                if start_date <= article_date <= end_date:
                    urls_to_process.append(article.get("url", ""))
            except json.JSONDecodeError:
                continue
    
    # Use multiprocessing to fetch articles and extract image URLs
    all_image_urls = []
    
    num_processes = cpu_count()
    with Pool(num_processes) as pool:
        image_lists = list(tqdm(
            pool.imap_unordered(extract_image_urls, urls_to_process),
            total=len(urls_to_process),
            desc="Processing popular articles"
        ))
    
    # Aggregate all image URLs
    for images in image_lists:
        all_image_urls.extend(images)
    
    # Output
    output_data = {
        "number_of_popular_articles": len(urls_to_process),
        "image_urls": all_image_urls
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Popular articles processed: {output_path}")


def extract_content_and_images(url):
    """Extract article content and image URLs. Returns None if "※ 發信站" marker not found."""
    session = requests.Session()
    session.cookies.update({"over18": "1"})
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    })
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        return None
    
    soup = BeautifulSoup(resp.text, "html.parser")
    main_content_div = soup.find("div", id="main-content")
    
    if not main_content_div:
        return None
    
    # First check if "※ 發信站" marker exists - if not, skip this article
    end_marker = None
    for span in main_content_div.find_all("span", class_="f2"):
        if "※ 發信站" in span.get_text():
            end_marker = span
            break
    
    if not end_marker:
        return None  # Skip this article if marker doesn't exist
    
    # Find the start: first metaline div with "作者"
    metalines = main_content_div.find_all("div", class_="article-metaline")
    start_elem = None
    
    if metalines:
        start_elem = metalines[0]
    else:
        return None
    
    # Extract text between start (inclusive) and end (exclusive)
    content = ""
    if start_elem:
        # Collect all text between start and end markers
        content_parts = []
        current = start_elem
        
        while current:
            if current == end_marker:
                break
            
            # Add text from current element
            if current.string:
                text = str(current.string).strip()
                if text:
                    content_parts.append(text)
            
            # Traverse to next element (depth-first, all descendants)
            if current.contents:
                # Has children - go to first child
                for child in current.children:
                    if isinstance(child, type(current)):  # It's an element
                        next_elem = child
                        break
                else:
                    # No element children, go to next sibling
                    next_elem = current.find_next_sibling()
            else:
                # No children, go to next sibling or find next via parent
                next_elem = current.find_next_sibling()
            
            if next_elem is None or next_elem == end_marker:
                break
            
            current = next_elem
        
        # Simpler approach: get all text from main-content, then extract between markers
        main_text = main_content_div.get_text()
        
        # Find where start marker text ends and end marker text begins
        start_text = start_elem.get_text()
        end_text = end_marker.get_text()
        
        # Extract content between them
        start_idx = main_text.find(start_text)
        end_idx = main_text.find(end_text, start_idx)
        
        if start_idx >= 0 and end_idx > start_idx:
            content = main_text[start_idx + len(start_text):end_idx].strip()
        elif start_idx >= 0:
            # If end marker not found, take everything from start
            content = main_text[start_idx + len(start_text):].strip()
    
    # Extract image URLs - handle query parameters and fragments
    image_urls = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".gif")
    
    def is_image_url(test_url):
        """Check if URL is an image URL, handling query params and fragments."""
        if not (test_url.startswith("http://") or test_url.startswith("https://")):
            return False
        # Remove query parameters and fragments
        url_path = test_url.split("?")[0].split("#")[0]
        return any(url_path.lower().endswith(ext) for ext in valid_extensions)
    
    # Check all <a> tags
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if is_image_url(href):
            image_urls.append(href)
    
    # Check all <img> tags
    for img_tag in soup.find_all("img", src=True):
        src = img_tag["src"]
        if is_image_url(src):
            image_urls.append(src)
    
    # Also extract bare URLs from text (image URLs in plain text like in comments)
    page_text = soup.get_text()
    url_pattern = r'https?://[^\s\)\]]*\.(?:jpg|jpeg|png|gif)'
    for match in re.finditer(url_pattern, page_text, re.IGNORECASE):
        matched_url = match.group(0)
        # Clean up any trailing punctuation
        matched_url = matched_url.rstrip('.,;:)')
        if is_image_url(matched_url):
            image_urls.append(matched_url)
    
    return {"content": content, "image_urls": image_urls}


def keyword(start_date, end_date, search_keyword):
    """Search for keyword in article content (from 作者 to 發信站) and extract image URLs."""
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    articles_path = os.path.join(output_dir, "articles.jsonl")
    output_path = os.path.join(output_dir, f"keyword_{start_date}_{end_date}_{search_keyword}.json")
    
    # Read articles from articles.jsonl and filter by date
    urls_to_process = []
    
    with open(articles_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
                article_date = article.get("date", "")
                
                # Check if date is within range
                if start_date <= article_date <= end_date:
                    url = article.get("url", "")
                    urls_to_process.append(url)
            except json.JSONDecodeError:
                continue
    
    # Process articles with multiprocessing
    all_image_urls = []
    
    num_processes = cpu_count()
    
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(extract_content_and_images, urls_to_process),
            total=len(urls_to_process),
            desc="Processing articles for keyword"
        ))
    
    # Check results and collect matching images
    for result in results:
        if result is None:
            continue
        
        # Check if keyword is in content (from 作者 to 發信站)
        if search_keyword in result["content"]:
            all_image_urls.extend(result["image_urls"])
    
    # Output
    output_data = {
        "image_urls": all_image_urls
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Keyword search completed: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HW4 PTT Stock Crawler")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    crawl_parser = subparsers.add_parser("crawl", help="Crawl 2025 articles")
    
    push_parser = subparsers.add_parser("push", help="Extract push/boo statistics")
    push_parser.add_argument("start_date", help="Start date (MMDD format)")
    push_parser.add_argument("end_date", help="End date (MMDD format)")
    
    popular_parser = subparsers.add_parser("popular", help="Extract images from popular articles")
    popular_parser.add_argument("start_date", help="Start date (MMDD format)")
    popular_parser.add_argument("end_date", help="End date (MMDD format)")
    
    keyword_parser = subparsers.add_parser("keyword", help="Search keyword and extract images")
    keyword_parser.add_argument("start_date", help="Start date (MMDD format)")
    keyword_parser.add_argument("end_date", help="End date (MMDD format)")
    keyword_parser.add_argument("search_keyword", help="Keyword to search")
    
    debug_keyword_parser = subparsers.add_parser("debug_keyword", help="Debug keyword search - print first 3 matching articles")
    debug_keyword_parser.add_argument("start_date", help="Start date (MMDD format)")
    debug_keyword_parser.add_argument("end_date", help="End date (MMDD format)")
    debug_keyword_parser.add_argument("search_keyword", help="Keyword to search")
    
    debug_parser = subparsers.add_parser("debug", help="Debug a single article")
    debug_parser.add_argument("url", help="Article URL to debug")

    args = parser.parse_args()

    if args.command == "crawl":
        crawl()
    elif args.command == "push":
        push(args.start_date, args.end_date)
    elif args.command == "popular":
        popular(args.start_date, args.end_date)
    elif args.command == "keyword":
        keyword(args.start_date, args.end_date, args.search_keyword)
    elif args.command == "debug_keyword":
        debug_keyword(args.start_date, args.end_date, args.search_keyword)
    elif args.command == "debug":
        debug_article(args.url)
    else:
        parser.print_help()


def debug_keyword(start_date, end_date, search_keyword):
    """Debug keyword search - print first 3 articles containing keyword."""
    output_dir = OUTPUT_DIR
    articles_path = os.path.join(output_dir, "articles.jsonl")
    
    # Read articles from articles.jsonl and filter by date
    urls_to_process = []
    
    with open(articles_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                article = json.loads(line)
                article_date = article.get("date", "")
                
                # Check if date is within range
                if start_date <= article_date <= end_date:
                    url = article.get("url", "")
                    urls_to_process.append((url, article.get("title", "")))
            except json.JSONDecodeError:
                continue
    
    print(f"Total articles in date range {start_date}-{end_date}: {len(urls_to_process)}")
    print(f"Searching for keyword: '{search_keyword}'\n")
    
    found_count = 0
    
    for url, title in urls_to_process:
        if found_count >= 3:
            break
        
        result = extract_content_and_images(url)
        
        if result is None:
            print(f"[SKIP] No marker - {title}")
            print(f"       URL: {url}\n")
            continue
        
        content = result["content"]
        image_urls = result["image_urls"]
        
        # Check if keyword is in content
        if search_keyword in content:
            found_count += 1
            print(f"[FOUND #{found_count}] {title}")
            print(f"       URL: {url}")
            print(f"       Content length: {len(content)} chars")
            print(f"       Image URLs found: {len(image_urls)}")
            print(f"       Content preview (first 200 chars):")
            print(f"       {content[:200]}")
            print(f"       Image URLs (first 5):")
            for i, img_url in enumerate(image_urls[:5]):
                print(f"         {i+1}. {img_url}")
            print()
        else:
            print(f"[NO MATCH] {title}")
            print(f"           Content length: {len(content)} chars")
            print(f"           Content preview: {content[:100]}")
            print()
    
    if found_count == 0:
        print(f"No articles found containing keyword '{search_keyword}'")
    else:
        print(f"\nTotal found: {found_count}/3")


def debug_article(url):
    """Debug a single article to inspect push/boo tags."""
    session = requests.Session()
    session.cookies.update({"over18": "1"})
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    })
    
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Error fetching: {exc}")
        return
    
    soup = BeautifulSoup(resp.text, "html.parser")
    push_divs = soup.find_all("div", class_="push")
    
    print(f"Found {len(push_divs)} push divs\n")
    
    # Categorize all entries
    push_count = 0
    boo_count = 0
    neutral_count = 0
    
    # Show first 15 entries
    for i, push_div in enumerate(push_divs[:15]):
        tag_span = push_div.find("span", class_=lambda x: x and "push-tag" in x)
        if not tag_span:
            continue
        
        tag_text = tag_span.get_text(strip=True)
        userid_span = push_div.find("span", class_=lambda x: x and "push-userid" in x)
        user_id = userid_span.get_text() if userid_span else "UNKNOWN"
        
        # Show character codes
        chars = [f"{c}(U+{ord(c):04X})" for c in tag_text]
        
        print(f"{i+1}. Tag: '{tag_text}' = {chars}")
        print(f"   User: {repr(user_id)}")
        
        # Classify
        if "推" in tag_text:
            print(f"   -> PUSH")
            push_count += 1
        elif any(c in tag_text for c in ["嘘", "噓"]):
            print(f"   -> BOO")
            boo_count += 1
        else:
            print(f"   -> NEUTRAL")
            neutral_count += 1
        print()
    
    print(f"\nSummary (first 15): {push_count} push, {boo_count} boo, {neutral_count} neutral")


if __name__ == "__main__":
    main()

import argparse
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import json
import os
import random
import time
from tqdm import tqdm

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

    if os.path.exists(output_path):
        os.remove(output_path)

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

    with open(output_path, "a", encoding="utf-8") as fout:
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

                article = {"date": date_mmdd, "title": title, "url": url}
                fout.write(json.dumps(article, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="HW4 PTT Stock Crawler")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    crawl_parser = subparsers.add_parser("crawl", help="Crawl 2025 articles")

    args = parser.parse_args()

    if args.command == "crawl":
        crawl()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

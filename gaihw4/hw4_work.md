# NYCU AI Homework 4 - Crawling Releases (PTT Stock Board 2025)

## Overview
Create a Python script (`{student_id}.py`) that crawls PTT Stock board articles from 2025 and supports 4 command-line functions: Crawl, Push, Popular, and Keyword search.

**Deadline:** 2026/05/19 23:59

---

## Core Requirements

### 1. **Crawl Function** (25% of grade, 20-minute time limit)
**Command:** `python {student_id}.py crawl`

**Input:** None

**Functionality:**
- Crawl all Stock board articles from 2025
- Starting point: First 2025 article "[心得] 2024年報" (https://www.ptt.cc/bbs/Stock/M.1735671336.A.BD5.html)
- **Filters (exclude):**
  - Articles with "[公告]" in title
  - Articles with "Fw:[公告]" in title
  - Articles without title or empty title
  - Articles without corresponding URL

**Output Files:** (in current directory)
1. `articles.jsonl` - All valid articles (JSONL format)
2. `popular_articles.jsonl` - Popular articles only (JSONL format)

**JSON Format:**
```json
{"date": "MMDD", "title": "{title}", "url": "https://..."}
```

**Key Notes:**
- Use title from listing page (not from article content page)
- URL must start with `https://`
- "推爆" (Popular) definition: (推文數 - 噓文數) > 100 times
- Article order does not affect scoring
- Use append mode to save progress continuously (avoid data loss on exceptions)

---

### 2. **Push Function** (25% of grade, 10-minute time limit per test)
**Command:** `python {student_id}.py push {start_date} {end_date}`

**Input:**
- `{start_date}`: Date range start (MMDD format, e.g., 0304 for March 4)
- `{end_date}`: Date range end (MMDD format, e.g., 1231 for December 31)

**Functionality:**
- Scrape all articles in the date range (inclusive) and compute push/boo statistics
- Identify top 10 users by push count
- Identify top 10 users by boo count

**Output File:** `push_{start_date}_{end_date}.json`

**JSON Format:**
```json
{
  "push": {
    "total": {count},
    "top10": [
      {"user_id": "{user_id}", "count": {count}},
      ...
    ]
  },
  "boo": {
    "total": {count},
    "top10": [
      {"user_id": "{user_id}", "count": {count}},
      ...
    ]
  }
}
```

**Key Notes:**
- 推文 = push (❤ or 推)
- 噓文 = boo (→ or 噓)
- 中立留言 = neutral comments (= or other symbols)
- Maximum date span: 92 days

---

### 3. **Popular Function** (25% of grade, 10-minute time limit per test)
**Command:** `python {student_id}.py popular {start_date} {end_date}`

**Input:**
- `{start_date}`: Date range start (MMDD format)
- `{end_date}`: Date range end (MMDD format)

**Functionality:**
- Count number of popular (推爆) articles in date range (inclusive)
- Extract all image URLs from popular articles
- Include images from both article content and comments

**Output File:** `popular_{start_date}_{end_date}.json`

**JSON Format:**
```json
{
  "number_of_popular_articles": {count},
  "image_urls": [
    "https://...",
    "http://...",
    ...
  ]
}
```

**Image URL Definition:**
- Must start with `http://` or `https://`
- Must end with `.jpg`, `.jpeg`, `.png`, or `.gif` (case-insensitive)
- Order does not matter
- No need to remove duplicates

**Key Notes:**
- Maximum date span: 92 days

---

### 4. **Keyword Function** (25% of grade, 10-minute time limit per test)
**Command:** `python {student_id}.py keyword {start_date} {end_date} {keyword}`

**Input:**
- `{start_date}`: Date range start (MMDD format)
- `{end_date}`: Date range end (MMDD format)
- `{keyword}`: Search keyword (no whitespace guaranteed)

**Functionality:**
- Search keyword in title first, then in article content if title does not match
- If a URL appears in content, follow it and search its content as well
- Exact word match only; do not attempt language normalization or fuzzy matching
- Extract all image URLs from matching articles
- Include images from article content and comments

**Content Range Definition:**
- Start: "作者" (author line, inclusive)
- End: "※ 發信站" (green info line, exclusive)
- If "※ 發信站" doesn't exist, ignore the article
- Note: Use article content title (not listing page title)

**Output File:** `keyword_{start_date}_{end_date}_{keyword}.json`

**JSON Format:**
```json
{
  "image_urls": [
    "https://...",
    "http://...",
    ...
  ]
}
```

**Key Notes:**
- Keyword matching includes URLs in the content
- Order does not matter
- No need to remove duplicates
- Maximum date span: 92 days

---

## Technical Specifications

### Testing Environment
- **OS:** Ubuntu 24.04
- **Python:** 3.12.3 (compatible with 3.12.x)
- **IP:** Independent IP from Engineering Building 4

### Allowed Packages
```
beautifulsoup4==4.13.4
click==8.1.8
html5lib==1.1
httpx==0.28.1
lxml==5.3.2
pandas==2.2.3
pyquery==2.0.1
requests==2.32.3
scrapy==2.12.0
tqdm==4.67.1
```

### Key Implementation Guidelines
1. **Multi-processing:** Use `os.cpu_count()` to determine number of processes
2. **Append Mode:** Use append mode when saving to handle exceptions gracefully
3. **Time Limits:** 
   - Crawl: 20 minutes
   - Push/Popular/Keyword: 10 minutes each
   - Programs exceeding time limit will be forcibly killed
4. **Date Format:** Always MMDD (e.g., 0101 for January 1, 1231 for December 31)

---

## Submission Requirements

### File Name
- **Single file only:** `{student_id}.py`
- Replace `{student_id}` with your actual student ID

### Scoring Breakdown
| Function | Points | Time Limit | Notes |
|----------|--------|-----------|-------|
| Crawl | 25% | 20 min | 1 test |
| Push | 25% | 10 min | 5 test cases (5 points each) |
| Popular | 25% | 10 min | 5 test cases (5 points each) |
| Keyword | 25% | 10 min | 5 test cases (5 points each) |

### Penalty
- Format error (but manually fixable): Score × 0.8
- Cannot execute: No credit

---

## Testing & Validation

### Provided Evaluation Tool
- **File:** `eval.py` (in hw4.zip)
- **Sample Answers:** `2025_answer/` folder (contains partial 2025 test cases)
- **Usage:** `python eval.py 2025_answer outputs`

### Output Directory Structure
```
outputs/
├── articles.jsonl
├── popular_articles.jsonl
├── push_MMDD_MMDD.json
├── popular_MMDD_MMDD.json
└── keyword_MMDD_MMDD_keyword.json
```

---

## Important Notes

1. **Data Consistency:** Results may change over time due to user push/boo/delete actions
   - 2025_answer may not be correct when you test
   - TA will generate fresh answers close to grading time

2. **Article Title Source:** Use listing page title (not article content page title)

3. **URL Format:** Always use `https://` (not `http://`)

4. **Push/Boo Symbols:**
   - 推 or ❤ = push
   - 噓 or → = boo
   - = or other = neutral

5. **Keyword Matching:** 
   - Case-sensitive (assuming Chinese keywords in examples)
   - Include URLs that contain the keyword
   - Content range: from author to green info line

---

## Clarified Requirements

1. **Push Function:** Process all scraped articles in the date range, then pick the first 10 users with the most push or boo counts.
2. **Popular Articles JSONL:** Each line must include only the publish date, title, and URL. No additional extraction required.
3. **Keyword Search:** Match keywords in title first. If no title hit, fetch the article source and search its content. If the article contains a URL, follow it and search that page too.
4. **Keyword Matching:** Use exact word matching only. Do not normalize languages or match similar words.
5. **Caching:** No persistent article cache is required; the command can fetch article content on demand.
6. **HTML Parsing for Content:** Focus on relevant HTML elements such as `<title>`, `<p>`, and `<a href="...">` when searching content.
7. **Multi-processing:** Apply multiprocessing to any scraping task that is time-consuming, including crawl, push, popular, and keyword commands.

---

## Remaining Questions

> If any requirement is still unclear, please point out the specific command or output format.


import json
import time
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "wikipedia_articles.jsonl"
TARGET_DOCUMENTS = 5000
API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "AITextPipelineStudentProject/1.0 (student educational project)"
}


def collect_articles():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = {}
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                    if record.get("id") and record.get("text"):
                        existing[str(record["id"])] = record
                except json.JSONDecodeError:
                    continue

    print(f"Existing documents: {len(existing)}")

    session = requests.Session()
    session.headers.update(HEADERS)
    continuation = {}
    failures = 0

    with OUTPUT_FILE.open("a", encoding="utf-8") as file:
        while len(existing) < TARGET_DOCUMENTS:
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": 10,
                "maxlag": 5,
            }
            params.update(continuation)

            try:
                response = session.get(
                    API_URL, params=params, timeout=30
                )

                if response.status_code in (429, 503):
                    wait = int(response.headers.get("Retry-After", 5))
                    print(f"Server busy. Waiting {wait} seconds...")
                    time.sleep(max(wait, 5))
                    continue

                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    print("Wikipedia API message:", data["error"])
                    time.sleep(5)
                    continue

                page_ids = [
                    str(page["id"])
                    for page in data.get("query", {}).get("random", [])
                    if str(page["id"]) not in existing
                ]

                if page_ids:
                    extract_params = {
                        "action": "query",
                        "format": "json",
                        "pageids": "|".join(page_ids),
                        "prop": "extracts",
                        "explaintext": 1,
                        "exchars": 20000,
                        "maxlag": 5,
                    }

                    extract_response = session.get(
                        API_URL, params=extract_params, timeout=30
                    )
                    extract_response.raise_for_status()
                    extract_data = extract_response.json()

                    for page in extract_data.get(
                        "query", {}
                    ).get("pages", {}).values():
                        title = page.get("title", "").strip()
                        text = page.get("extract", "").strip()
                        page_id = str(page.get("pageid", ""))

                        if (
                            not page_id
                            or page_id in existing
                            or not title
                            or len(text) < 100
                        ):
                            continue

                        record = {
                            "id": page_id,
                            "title": title,
                            "text": text,
                            "source": "Wikipedia",
                            "url": f"https://en.wikipedia.org/?curid={page_id}",
                        }

                        file.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                        file.flush()
                        existing[page_id] = record

                        if len(existing) % 100 == 0:
                            print(
                                f"Collected {len(existing)} unique documents"
                            )

                        if len(existing) >= TARGET_DOCUMENTS:
                            break

                continuation = data.get("continue", {})
                failures = 0
                time.sleep(0.5)

            except (requests.RequestException, ValueError) as error:
                failures += 1
                wait = min(5 * failures, 60)
                print(f"Request failed: {error}")
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)

                if failures >= 10:
                    print("Too many consecutive failures. Stopping safely.")
                    break

    print(f"\nTotal unique documents: {len(existing)}")
    print(f"Output file: {OUTPUT_FILE}")

    if len(existing) >= TARGET_DOCUMENTS:
        print("SUCCESS: Dataset target reached!")
    else:
        print("Target not reached yet. Run the script again to resume.")


if __name__ == "__main__":
    collect_articles()

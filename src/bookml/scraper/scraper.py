"""
Scrape Fiction and Nonfiction books from https://books.toscrape.com/
Category listing (page 1) -> individual product pages (page 2) -> CSV.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = "https://books.toscrape.com/"
TARGET_CATEGORIES = ("Fiction", "Nonfiction")
DEFAULT_OUTPUT = "src/bookml/data/books_all.csv"
DEFAULT_IMAGES_DIR = (
    Path("src/bookml/data/images")
    if Path("src/bookml").exists()
    else Path("data/images")
)
WAIT_SECONDS = 15


def create_driver(headless: bool = True) -> webdriver.Edge:
    options = EdgeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Edge(options=options)


def wait_for(driver: webdriver.Edge) -> WebDriverWait:
    return WebDriverWait(driver, WAIT_SECONDS)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def get_category_urls(driver: webdriver.Edge) -> dict[str, str]:
    """Page 1: discover Fiction and Nonfiction links from the sidebar."""
    driver.get(BASE_URL)
    wait_for(driver).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.side_categories"))
    )
    urls: dict[str, str] = {}
    for name in TARGET_CATEGORIES:
        link = driver.find_element(
            By.XPATH,
            f"//div[@class='side_categories']//a[normalize-space(text())='{name}']",
        )
        urls[name] = urljoin(BASE_URL, link.get_attribute("href"))
    return urls


def collect_book_urls_from_category(
    driver: webdriver.Edge, category_url: str
) -> list[str]:
    """All product page URLs for one category, following pagination."""
    driver.get(category_url)
    wait_for(driver).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "article.product_pod"))
    )

    book_urls: list[str] = []
    seen_pages: set[str] = set()

    while True:
        current = driver.current_url
        if current in seen_pages:
            break
        seen_pages.add(current)

        wait_for(driver).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.product_pod"))
        )
        for anchor in driver.find_elements(
            By.CSS_SELECTOR, "article.product_pod h3 a"
        ):
            href = anchor.get_attribute("href")
            if href:
                book_urls.append(urljoin(BASE_URL, href))

        next_links = driver.find_elements(By.CSS_SELECTOR, "li.next > a")
        if not next_links:
            break
        next_href = next_links[0].get_attribute("href")
        if not next_href:
            break
        driver.get(urljoin(BASE_URL, next_href))

    return book_urls


def scrape_product_page(driver: webdriver.Edge, category: str) -> dict[str, str]:
    """Page 2: extract all fields from the current product detail page."""
    wait_for(driver).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.product_main h1"))
    )

    row: dict[str, str] = {"category": category}

    row["title"] = normalize_text(
        driver.find_element(By.CSS_SELECTOR, "div.product_main h1").text
    )
    row["price"] = normalize_text(
        driver.find_element(By.CSS_SELECTOR, "p.price_color").text
    )

    availability_el = driver.find_elements(
        By.CSS_SELECTOR, "p.instock.availability"
    )
    row["availability_summary"] = (
        normalize_text(availability_el[0].text) if availability_el else ""
    )

    rating_el = driver.find_elements(By.CSS_SELECTOR, "p.star-rating")
    if rating_el:
        classes = rating_el[0].get_attribute("class") or ""
        parts = classes.split()
        row["rating"] = parts[1] if len(parts) > 1 else ""
    else:
        row["rating"] = ""

    breadcrumb = driver.find_elements(By.CSS_SELECTOR, "ul.breadcrumb li")
    row["breadcrumb"] = " / ".join(
        normalize_text(item.text) for item in breadcrumb if normalize_text(item.text)
    )

    image_el = driver.find_elements(
        By.CSS_SELECTOR, "#product_gallery img, div.product_gallery img"
    )
    row["image_url"] = image_el[0].get_attribute("src") if image_el else ""

    desc_heading = driver.find_elements(By.ID, "product_description")
    if desc_heading:
        sibling = driver.execute_script(
            "return arguments[0].nextElementSibling;", desc_heading[0]
        )
        row["description"] = (
            normalize_text(sibling.text) if sibling is not None else ""
        )
    else:
        row["description"] = ""

    for tr in driver.find_elements(By.CSS_SELECTOR, "table.table.table-striped tr"):
        headers = tr.find_elements(By.TAG_NAME, "th")
        cells = tr.find_elements(By.TAG_NAME, "td")
        if headers and cells:
            key = normalize_text(headers[0].text).lower()
            key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
            row[key] = normalize_text(cells[0].text)

    return row


def get_image_filename(title: str, category: str) -> str:
    norm_title = title.replace("11/22/63", "11/22/1963")
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", norm_title)[:50]
    return f"{category}_{sanitized}.jpg"


def download_image(image_url: str, dest_path: Path, timeout: int = 15) -> bool:
    if not image_url:
        return False
    try:
        req = Request(image_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urlopen(req, timeout=timeout) as response:
            data = response.read()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  Failed to download image {image_url}: {exc}", file=sys.stderr)
        return False


def write_csv(rows: list[dict[str, str]], path: str) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "category",
        "title",
        "price",
        "rating",
        "availability_summary",
        "description",
        "upc",
        "product_type",
        "price_excl_tax",
        "price_incl_tax",
        "tax",
        "availability",
        "number_of_reviews",
        "breadcrumb",
        "image_url",
        "image_file",
        "url",
    ]
    for name in preferred:
        if name not in seen:
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scrape(
    output_path: str = DEFAULT_OUTPUT,
    images_dir: Path | str = DEFAULT_IMAGES_DIR,
    download_images: bool = True,
):
    driver = create_driver()
    all_rows: list[dict[str, str]] = []

    img_dest_dir = Path(images_dir)
    if download_images:
        img_dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        category_urls = get_category_urls(driver)
        print(f"Categories: {', '.join(category_urls.keys())}")

        for category, category_url in category_urls.items():
            book_urls = collect_book_urls_from_category(driver, category_url)
            print(f"{category}: {len(book_urls)} books")

            for index, book_url in enumerate(book_urls, start=1):
                driver.get(book_url)
                try:
                    row = scrape_product_page(driver, category)
                    row["url"] = book_url

                    # Download book cover image using standardized naming convention
                    if download_images and row.get("image_url"):
                        img_filename = get_image_filename(row.get("title", ""), category)
                        img_path = img_dest_dir / img_filename
                        download_image(row["image_url"], img_path)
                        row["image_file"] = img_filename

                    all_rows.append(row)
                except TimeoutException:
                    print(f"  Skipped (timeout): {book_url}", file=sys.stderr)
                if index % 20 == 0:
                    print(f"  {category}: {index}/{len(book_urls)} scraped")

        write_csv(all_rows, output_path)
        print(f"Wrote {len(all_rows)} rows to {output_path}")
        if download_images:
            print(f"Saved cover images to {img_dest_dir}")

    finally:
        driver.quit()



if __name__ == "__main__":
    scrape()

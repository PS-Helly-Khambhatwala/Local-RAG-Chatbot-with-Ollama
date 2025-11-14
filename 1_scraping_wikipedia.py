from bs4 import BeautifulSoup
import requests
import pandas as pd
import os

pd.options.mode.chained_assignment = None

keywords = pd.read_excel("keywords.xlsx")

print("Loaded keywords:")
print(keywords.head())

#SETUP OUTPUT FOLDER
DATASET_FOLDER = "data/"
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

#SCRAPE WIKIPEDIA PAGES
def scrape_wikipedia(keyword):
    url = f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}"
    print(f"Scraping: {url}")

    #added headers to look like normal user beacuse wikipedia blocks robots
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch {url} (status {response.status_code})")
        return None

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text() for p in paragraphs)

    return text.strip()

# LOOP THROUGH KEYWORDS AND SAVE DATA
for ind in keywords.index:
    keyword = str(keywords.loc[ind, "Keyword"])
    print(f" Processing keyword: {keyword}")

    content = scrape_wikipedia(keyword)
    if content:
        file_path = os.path.join(DATASET_FOLDER, f"{keyword.replace(' ', '_')}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved: {file_path}")
    else:
        print(f"Skipped: {keyword}")

print("\nDone! All Wikipedia pages saved in 'data/' folder.")

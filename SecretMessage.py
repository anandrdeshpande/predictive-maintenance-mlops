"""
Secret Message Grid Printer
============================

Problem
-------
Given the URL of a published Google Doc that contains a table with three
columns (x-coordinate, Unicode character, y-coordinate), retrieve the
document, parse the table, and print a 2D grid of characters. When printed
in a fixed-width font, the populated cells form a graphic spelling out a
secret message in uppercase letters.

Coordinate system
------------------
(0, 0) is the TOP-LEFT corner of the grid.
  - x increases to the RIGHT  (column index)
  - y increases DOWNWARD      (row index)

Usage
-----
    from print_secret_message import print_secret_message
    print_secret_message("https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub")

Or from the command line:
    python print_secret_message.py "<google-doc-url>"
"""

import re
import sys
import requests
from bs4 import BeautifulSoup


def _to_published_html_url(url: str) -> str:
    """
    Normalize a Google Doc URL so we request the plain HTML "published to
    web" version, which is easy to parse with BeautifulSoup and doesn't
    require authentication.

    Handles:
      - Already-published URLs ending in /pub or /pub?...       -> used as-is
      - Normal edit URLs: .../document/d/<ID>/edit...            -> rewritten
      - Bare IDs                                                 -> rewritten
    """
    # Already looks like a published doc URL.
    if "/pub" in url:
        return url

    # Try to pull the document ID out of a standard share/edit URL.
    match = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
    if match:
        doc_id = match.group(1)
        return f"https://docs.google.com/document/d/{doc_id}/pub"

    # Fall back: assume the whole string is already a doc ID.
    return f"https://docs.google.com/document/d/{url}/pub"



def _fetch_grid_data(url: str):
    """
    Download the document and parse the (x, char, y) triples out of its
    table. Returns a list of (x, y, char) tuples.
    """
    fetch_url = _to_published_html_url(url)

    resp = requests.get(fetch_url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError(
            "No table found in the document. Make sure the URL points to "
            "a published Google Doc containing the coordinate table."
        )

    rows = table.find_all("tr")

    entries = []
    # Skip the header row (first row: "x-coordinate", "Character", "y-coordinate")
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        x_text = cells[0].get_text(strip=True)
        char_text = cells[1].get_text(strip=True)
        y_text = cells[2].get_text(strip=True)

        # Skip any stray empty rows.
        if not x_text or not y_text or not char_text:
            continue

        try:
            x = int(x_text)
            y = int(y_text)
        except ValueError:
            continue

        entries.append((x, y, char_text))

    if not entries:
        raise ValueError("Table was found but no valid (x, char, y) rows were parsed.")

    return entries


def print_secret_message(document_url: str) -> None:
    """
    Retrieve and parse the grid data from the given Google Doc URL, then
    print the resulting character grid to stdout. This is the single
    required entry-point function described by the exercise spec.
    """
    entries = _fetch_grid_data(document_url)

    max_x = max(x for x, y, ch in entries)
    max_y = max(y for x, y, ch in entries)

    # Build a blank grid filled with spaces: grid[y][x]
    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for x, y, ch in entries:
        grid[y][x] = ch

    for row in grid:
        print("".join(row))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_secret_message.py <google-doc-url>")
        sys.exit(1)

    print_secret_message(sys.argv[1])
    
    #print_secret_message("https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub")
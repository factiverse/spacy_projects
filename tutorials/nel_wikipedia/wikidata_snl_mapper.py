import codecs
import json
from pathlib import Path
from typing import Optional, Text

import bs4
import requests

from wikidata_utils import Wikidata


def get_soup(url: str, parser: str = "html.parser") -> Optional[bs4.BeautifulSoup]:
    """Get soup for the url.

    Args:
        url: url to parse with BeautifulSoup.
        parser: parser to use. Let's you use function for other parsers as well.
    Returns:
        BeautifulSoup instance that has parsed the content.
    """
    try:
        req = requests.get(url)
        if req.status_code != 200:
            raise ValueError(f"Failed to request {url}")
        soup = bs4.BeautifulSoup(req.text, parser)
        return soup
    except Exception as e:
        print(e)
        return None


def get_snl_article_id(url: Text) -> int:
    soup = get_soup(url)
    if not soup:
        return -1
    scripts = soup.find("script").string
    scripts = scripts.replace("\n", " ").replace("'", '"')
    data_layer = json.loads(
        scripts[scripts.find("[") + 1 : scripts.find("]")]  # noqa: E203
    )
    return data_layer.get("articleId")


if __name__ == "__main__":
    data_path = "/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_no_kb_train_output_250922"
    wd = Wikidata()
    with codecs.open(Path(data_path) / "snl_ids.tsv", "w", "utf-8") as outf:
        with open(Path(data_path) / "entity_defs.csv") as f:
            next(f)
            for line in f:
                [wikidata_label, wikidata_id] = line.split("|")
                wikidata_id = wikidata_id.strip("\n")
                if wikidata_id:
                    snl_url = wd.get_snl_id(wikidata_id)
                    if snl_url:
                        snl_article_id = get_snl_article_id(snl_url)
                        outf.write(
                            f"{wikidata_id}\t{wikidata_label}\t{snl_url}\t{snl_article_id}\n"
                        )

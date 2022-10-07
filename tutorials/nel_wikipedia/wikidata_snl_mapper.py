import argparse
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
    parser = argparse.ArgumentParser("Create Spacy KB")
    parser.add_argument(
        "--kb_dir",
        type=str,
        required=True,
        help="Director containing KB",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_no_kb_train_output_250922",
    )
    args = parser.parse_args()
    data_path = args.kb_dir
    wd = Wikidata()
    with codecs.open(Path(data_path) / "snl_ids_article_ids.tsv", "w", "utf-8") as outf:
        with open(Path(data_path) / "snl_ids.tsv") as f:
            next(f)
            for line in f:
                [wikidata_url, snl_id] = line.split("\t")
                wikidata_id = wikidata_url.split("/")[-1]
                snl_url = f"https://snl.no/{snl_id}"
                print(wikidata_id, snl_id)
                if wikidata_id:
                    if snl_url:
                        snl_article_id = get_snl_article_id(snl_url)
                        if snl_article_id != -1:
                            outf.write(f"{wikidata_id}\t{snl_url}\t{snl_article_id}\n")

import argparse
from pathlib import Path

import spacy

from entity_linker_trainer import read_el_docs_golds
from kb_creator import read_kb
from nel_evaluation import measure_performance


def parse_args():
    parser = argparse.ArgumentParser("NEL evaluator")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Model name or path",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_no_kb_train_output_250922_spacy3_kb_nel_full_iter200/trained_nel/",
    )
    parser.add_argument(
        "--kb_dir",
        type=str,
        required=False,
        help="Director containing KB",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_no_kb_train_output_250922_spacy3_kb/kb",
    )
    parser.add_argument(
        "--gold_file",
        type=str,
        required=False,
        help="Director containing gold json",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922/gold_entities_shuffled.jsonl",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        required=False,
        help="Number of train docs",
        default=100,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("loading model")
    nlp = spacy.load(args.model)
    print("loading KB")
    kb = read_kb(Path(args.kb_dir), Path(args.model))

    dev_docs = read_el_docs_golds(
        kb=kb,
        nlp=nlp,
        entity_file_path=Path(args.gold_file),
        dev=True,
        dev_num_records=args.test_size,
    )
    print("Loaded dev data")
    measure_performance(dev_data=dev_docs, el_pipe=nlp, kb=kb)

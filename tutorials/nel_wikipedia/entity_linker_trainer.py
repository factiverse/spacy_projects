from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from asyncio.log import logger
from pathlib import Path
from typing import Any, List

import spacy
from spacy.ml.models import load_kb
from spacy.training import Example
from spacy.util import compounding, minibatch
from tqdm import tqdm

import kb_creator
from nel_evaluation import measure_performance

spacy.require_gpu()
spacy.prefer_gpu()


def is_valid_article(doc_text):
    # custom length cut-off
    return 10 < len(doc_text) < 30000


def is_valid_sentence(sent_text):
    if not 10 < len(sent_text) < 3000:
        # custom length cut-off
        return False

    if sent_text.strip().startswith("*") or sent_text.strip().startswith("#"):
        # remove 'enumeration' sentences (occurs often on Wikipedia)
        return False

    return True


def read_el_docs_golds(
    nlp,
    entity_file_path,
    kb,
    train_num_records=10000,
    dev_num_records=100,
    labels_discard=None,
    dev=False,
    translate_snl_ids=False,
    snl_id_dict={},
):
    """This method provides training/dev examples that correspond to the entity annotations found by the nlp object.
    For training, it will include both positive and negative examples by using the candidate generator from the kb.
    For testing (kb=None), it will include all positive examples only."""
    if not labels_discard:
        labels_discard = []
    dev_recs = 0
    train_recs = 0
    with entity_file_path.open("r", encoding="utf8") as _file:
        for i, line in enumerate(tqdm(_file)):
            # print(line)
            # if dev:
            #     if i < train_num_records:
            #         continue
            #     if dev_recs > dev_num_records:
            #         break
            if not dev and train_recs > train_num_records:
                break

            example = json.loads(line)
            article_id = example["article_id"]
            clean_text = example["clean_text"]
            entities = example["entities"]

            if not is_valid_article(clean_text):
                continue
            # print(clean_text)
            try:
                doc = nlp(clean_text)
            except Exception as e:
                logger.error(f"Failed to parse {clean_text}")
                continue
            gold = _get_gold_parse(
                doc,
                entities,
                dev=dev,
                kb=kb,
                labels_discard=labels_discard,
                translate_snl_ids=translate_snl_ids,
                snl_id_dict=snl_id_dict,
            )
            if gold and len(gold["entities"]) > 0:
                yield (str(doc), gold)
            line = _file.readline()
            train_recs += 1
            dev_recs += 1


def _get_gold_parse(
    doc, entities, dev, kb, labels_discard, translate_snl_ids, snl_id_dict
):
    all_links = {}
    all_annotations = []
    tagged_ent_positions = {
        (ent.start_char, ent.end_char): (ent, ent.label_)
        for ent in doc.ents
        if ent.label_ not in labels_discard
    }

    for entity in entities:
        entity_id = entity["entity"]
        if translate_snl_ids:
            if int(entity_id) in snl_id_dict:
                entity_id_tuple = snl_id_dict.get(int(entity_id))
                entity_id = entity_id_tuple[0]
            else:
                continue
        alias = entity["alias"]
        start = entity["start"]
        end = entity["end"]

        candidate_ids = []
        if kb and not dev:
            candidates = kb.get_alias_candidates(alias)
            candidate_ids = [cand.entity_ for cand in candidates]

        (tagged_ent, ent_type) = tagged_ent_positions.get((start, end), (None, None))
        if tagged_ent:
            # TODO: check that alias == doc.text[start:end]
            should_add_ent = (dev or entity_id in candidate_ids) and is_valid_sentence(
                tagged_ent.sent.text
            )

            if should_add_ent:
                value_by_id = {entity_id: 1.0}
                if not dev:
                    random.shuffle(candidate_ids)
                    value_by_id.update(
                        {kb_id: 0.0 for kb_id in candidate_ids if kb_id != entity_id}
                    )
                all_links[(start, end)] = value_by_id
                all_annotations.append((start, end, ent_type))
    return {"links": all_links, "entities": all_annotations}


def get_entity_annotations(entities):
    annotations = {}
    for entity in entities:
        QID = entity["entity"]

        offset = (entity["start"], entity["end"])

        alias = entity["alias"]
        annotations[offset] = (QID, alias)
    return annotations


def get_gold_ids(dataset):
    gold_ids = []
    for text, annot in dataset:
        for span, links_dict in annot["links"].items():
            for link, value in links_dict.items():
                if value:
                    gold_ids.append(link)

    return gold_ids


def get_train_examples(dataset, nlp):
    TRAIN_EXAMPLES = []
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    sentencizer = nlp.get_pipe("sentencizer")
    for text, annotation in dataset:
        example = Example.from_dict(nlp.make_doc(str(text)), annotation)
        example.reference = sentencizer(example.reference)
        TRAIN_EXAMPLES.append(example)
    return TRAIN_EXAMPLES


def create_entity_linker(nlp, train_examples: List[Any], dir: Path):
    entity_linker = nlp.add_pipe(
        "entity_linker", config={"incl_prior": True}, last=True
    )
    entity_linker.initialize(
        get_examples=lambda: train_examples, kb_loader=load_kb(dir / "my_kb")
    )
    return entity_linker


def train_entity_linker(
    nlp,
    # dev_data,
    kb,
    train_examples: List[Any],
    kb_path: Path,
    num_iter: int = 500,
    output_dir: Path = Path("."),
):
    entity_linker = nlp.add_pipe(
        "entity_linker", config={"incl_prior": True}, last=True
    )
    entity_linker.initialize(
        get_examples=lambda: train_examples, kb_loader=load_kb(kb_path)
    )
    with nlp.select_pipes(enable=["entity_linker"]):  # train only the entity_linker
        optimizer = nlp.resume_training()
        optimizer.learn_rate = 0.0005
        best_loss = 65536
        for itn in range(num_iter):  # 500 iterations takes about a minute to train
            random.shuffle(train_examples)
            batches = minibatch(
                # train_examples, size=compounding(128.0, 2048.0, 2.001)
                train_examples,
                size=256,
            )  # increasing batch sizes
            losses = {}

            for batch in batches:
                nlp.update(
                    batch,
                    drop=0.3,  # prevent overfitting
                    losses=losses,
                    sgd=optimizer,
                )
            print(itn, len(batch), "Losses", losses)
            if itn % 50 == 0:
                # print the training loss
                new_loss = losses["entity_linker"]
                # measure_performance(dev_data=dev_data, el_pipe=nlp, kb=kb)
                if new_loss <= best_loss:
                    print("Best loss so far dumping model")
                    nlp.to_disk(Path(output_dir) / "trained_nel/best_model")
                    best_loss = new_loss
                    # nlp_loaded = spacy.load(
                    #     Path(args.output_dir) / "trained_nel/best_model"
                    # )
                    # measure_performance(dev_data=dev_data, el_pipe=nlp_loaded, kb=kb)
    print(itn, "Losses", losses)


def parse_args():
    parser = argparse.ArgumentParser("Entity linker trainer")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
        default="/local/home/vsetty/spacy_nel/data/spacy3_en_entitylinker",
    )

    parser.add_argument(
        "--kb_dir",
        type=str,
        required=True,
        help="Director containing KB",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922",
    )

    parser.add_argument(
        "--gold_dir",
        type=str,
        required=True,
        help="Director containing gold entities json",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922",
    )
    parser.add_argument(
        "--iter", type=int, required=True, help="Number of iterations to run", default=1
    )
    parser.add_argument(
        "--train_size",
        type=int,
        required=False,
        help="Number of train docs",
        default=50000,
    )
    parser.add_argument(
        "-l",
        "--labels_exclude",
        nargs="+",
        help="Set of labels to exclude",
        required=False,
    )
    parser.add_argument("--transformer", action="store_true", required=False)
    parser.add_argument(
        "--no-transformer", dest="transformer", action="store_false", required=False
    )
    parser.set_defaults(transformer=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kb_dir = Path(args.kb_dir)
    print("Loading the spacy model")
    if args.transformer:
        custom_config = {
            "components": {
                "transformer": {
                    "model": {
                        "tokenizer_config": {
                            "use_fast": False,
                            "padding": True,
                            "add_special_tokens": True,
                            "truncation": True,
                        }
                    }
                }
            }
        }
        nlp = spacy.load(kb_dir / "nlp_kb", config=custom_config)
    else:
        nlp = spacy.load(kb_dir / "nlp_kb")
    # nlp = spacy.load(kb_dir / "nlp_kb")

    # nlp = spacy.load(
    #     "/local/home/vsetty/spacy_nel/data/nb_bert_ner/model-best/",
    #     config=custom_config,
    # )
    kb_path = Path(kb_dir / "kb")
    print("Loading KB")
    kb = kb_creator.read_kb(kb_path, kb_dir / "nlp_kb")
    print(len(kb))
    print(kb.entity_vector_length)
    print(kb.get_size_aliases())
    entity_file_path = Path(args.gold_dir) / "gold_entities_shuffled.jsonl"
    print(args.labels_exclude)

    if not os.path.exists(Path(args.output_dir)):
        os.makedirs(Path(args.output_dir))
        os.makedirs(Path(args.output_dir) / "trained_nel")

    if not os.path.exists(Path(args.output_dir) / "spacy_train_examples.pkl"):
        if not os.path.exists(Path(args.output_dir) / "dataset.pkl"):
            dataset = [
                gold
                for gold in read_el_docs_golds(
                    nlp=nlp,
                    entity_file_path=entity_file_path,
                    kb=kb,
                    dev=False,
                    train_num_records=args.train_size,
                    labels_discard=args.labels_exclude,
                )
            ]
            with open(Path(args.output_dir) / "dataset.pkl", "wb") as file:
                pickle.dump(dataset, file)
        else:
            with open(Path(args.output_dir) / "dataset.pkl", "rb") as file:
                dataset = pickle.load(file)
        print(dataset[0])
        train_examples = get_train_examples(dataset, nlp)
        with open(Path(args.output_dir) / "spacy_train_examples.pkl", "wb") as file:
            pickle.dump(train_examples, file)
    else:
        logger.info("spacy_train_examples.pkl exists loading it.")
        print("spacy_train_examples.pkl exists loading it.")
        with open(Path(args.output_dir) / "spacy_train_examples.pkl", "rb") as file:
            train_examples = pickle.load(file)
    # dev_data = [
    #     data
    #     for data in read_el_docs_golds(
    #         kb=kb,
    #         nlp=nlp,
    #         entity_file_path=Path(args.gold_dir) / "gold_entities_shuffled.jsonl",
    #         dev=True,
    #         dev_num_records=100,
    #         train_num_records=args.train_size,
    #     )
    # ]
    # print("Dev data loaded", len(dev_data))
    train_entity_linker(
        nlp=nlp,
        # dev_data=dev_data,
        kb=kb,
        train_examples=train_examples,
        kb_path=kb_path,
        num_iter=args.iter,
        output_dir=Path(args.output_dir),
    )
    text = "Med USAs historie menes gjerne dette områdets nyere historie, det vil si fra og med vestlige sjøfarere oppdaget de amerikanske kontinentene og koloniserte dem, blant annet den delen som senere skulle bli USA."
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)
    nlp.to_disk(Path(args.output_dir) / "trained_nel")
    nlp = spacy.load(Path(args.output_dir) / "trained_nel")
    measure_performance(
        dev_data=read_el_docs_golds(
            kb=kb,
            nlp=nlp,
            entity_file_path=Path(args.gold_dir) / "gold_entities_shuffled.jsonl",
            dev=True,
            dev_num_records=100,
            train_num_records=args.train_size,
        ),
        el_pipe=nlp,
        kb=kb,
    )
# CUDA_VISIBLE_DEVICES=1 python entity_linker_trainer.py --output_dir /local/home/vsetty/spacy_nel/data/en_core_web_lg/ --kb_dir /local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922_spacy3_en_kb --gold_dir /local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922 --train_size 50000 --no-transformer ---labels_exclude CARDINAL DATE ORDINAL PERCENT QUANTITY TIME
# CUDA_VISIBLE_DEVICES=1 python entity_linker_trainer.py --output_dir /local/home/vsetty/spacy_nel/data/paraphrase-multilingual-MiniLM-L12-v2/ --kb_dir /local/home/vsetty/spacy_nel/data/paraphrase-multilingual-MiniLM-L12-v2/trained_kb --gold_dir /local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_no_kb_train_output_250922 --train_size 10000 --iter 100

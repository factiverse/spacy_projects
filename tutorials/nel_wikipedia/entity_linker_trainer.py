from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import spacy
from spacy.ml.models import load_kb
from spacy.training import Example
from spacy.util import compounding, minibatch
from tqdm import tqdm

import kb_creator

spacy.require_gpu()

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

def read_el_docs_golds(nlp, entity_file_path, dev, kb, labels_discard=None):
    """ This method provides training/dev examples that correspond to the entity annotations found by the nlp object.
     For training, it will include both positive and negative examples by using the candidate generator from the kb.
     For testing (kb=None), it will include all positive examples only."""
    if not labels_discard:
        labels_discard = []

    

    with entity_file_path.open("r", encoding="utf8") as _file:
        for i, line in enumerate(tqdm(_file)):
            # print(line)
            if i > 10000:
               break
            example = json.loads(line)
            article_id = example["article_id"]
            clean_text = example["clean_text"]
            entities = example["entities"]

            if not is_valid_article(clean_text):
                continue

            doc = nlp(clean_text)
            gold = _get_gold_parse(
                doc, entities, dev=dev, kb=kb, labels_discard=labels_discard
            )
            if gold and len(gold["entities"]) > 0:
                yield (str(doc), gold)
            line = _file.readline()
            

def _get_gold_parse(doc, entities, dev, kb, labels_discard):
    all_links = {}
    all_annotations = []
    tagged_ent_positions = {
        (ent.start_char, ent.end_char): (ent, ent.label_)
        for ent in doc.ents
        if ent.label_ not in labels_discard
    }

    for entity in entities:
        entity_id = entity["entity"]
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
    
def load_gold_dataset(data_path: Path, nlp, kb):
    dataset = []
    json_loc = data_path / "gold_entities.jsonl"
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            text = example["clean_text"]
            doc = nlp(text)
            mentions = {}
            for ent in doc.ents:
                offset = (ent.start_char, ent.end_char)
                mentions[offset] = ent.label_
            # print(mentions)
            if len(mentions) == 0:
                continue
            entities = example["entities"]
            
            annotations = get_entity_annotations(entities)
            all_links = {}
            all_annotations = []
            for mention in mentions.items():
                (sent_span, entity_type) = mention
                absolute_span = (sent_span[0], sent_span[1])
                if absolute_span in annotations.keys():
                    links_dict = {}
                    entity_id, alias = annotations[absolute_span]
                    links_dict[entity_id] = 1.0
                    all_annotations.append((sent_span[0], sent_span[1], entity_type))
                    for candidate in kb.get_alias_candidates(alias):
                        if candidate.entity_ != entity_id:
                            links_dict[candidate.entity_] = 0.0
                    all_links[sent_span] = links_dict
            if len(all_links) > 0 and len(all_annotations) > 0:
                dataset.append((text, {"links": all_links, "entities": all_annotations}))
            if len(dataset) == 100:
                break
    return dataset

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
    entity_linker = nlp.add_pipe("entity_linker", config={"incl_prior": True}, last=True)
    entity_linker.initialize(get_examples=lambda: train_examples, kb_loader=load_kb(dir / "my_kb"))
    return entity_linker

def train_entity_linker(nlp, train_examples: List[Any], kb_path: Path, num_iter: int=500):
    entity_linker = nlp.add_pipe("entity_linker", config={"incl_prior": True}, last=True)
    entity_linker.initialize(get_examples=lambda: train_examples, kb_loader=load_kb(kb_path))
    with nlp.select_pipes(enable=["entity_linker"]):   # train only the entity_linker
        optimizer = nlp.resume_training()
        for itn in range(num_iter):   # 500 iterations takes about a minute to train
            random.shuffle(train_examples)
            batches = minibatch(train_examples, size=compounding(16.0, 128.0, 4.001))  # increasing batch sizes
            losses = {}
            for batch in batches:
                nlp.update(
                    batch,   
                    drop=0.5,      # prevent overfitting
                    losses=losses,
                    sgd=optimizer,
                )
            if itn % 50 == 0:
                print(itn, "Losses", losses)   # print the training loss
    print(itn, "Losses", losses)

def parse_args():
    parser = argparse.ArgumentParser("logically scraping instance.")
    parser.add_argument("--output_dir", type=str,
        required=True,
        help="Output directory",
        default="/local/home/vsetty/spacy_nel/data/spacy3_en_entitylinker")

    parser.add_argument("--kb_dir", type=str,
        required=True,
        help="Director containing KB",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922")
    
    parser.add_argument("--gold_dir", type=str,
        required=True,
        help="Director containing KB",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922")
    parser.add_argument("--iter", type=int,
        required=True,
        help="Number of iterations to run",
        default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    kb_dir = Path(args.kb_dir)
    nlp = spacy.load(kb_dir / "nlp_kb")
    kb_path = Path(kb_dir / "kb")
    kb = kb_creator.read_kb(kb_path, kb_dir/"nlp_kb")
    print(len(kb))
    print(kb.entity_vector_length)
    print(kb.get_size_aliases())
    entity_file_path = Path(args.gold_dir) / "gold_entities_shuffled.jsonl"
    dataset = [gold for gold in read_el_docs_golds(nlp=nlp, entity_file_path=entity_file_path, kb=kb, dev=False)]
    with open(Path(args.output_dir)/"dataset.pkl", 'wb') as file:
        pickle.dump(dataset, file)
    # dataset = load_gold_dataset(data_path=Path("/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922/"), nlp=nlp, kb=kb)
    print(dataset[0])
    train_examples = get_train_examples(dataset, nlp)
    train_entity_linker(nlp, train_examples=train_examples, kb_path=kb_path, num_iter=args.iter)
    text = "Med USAs historie menes gjerne dette områdets nyere historie, det vil si fra og med vestlige sjøfarere oppdaget de amerikanske kontinentene og koloniserte dem, blant annet den delen som senere skulle bli USA."
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)
    nlp.to_disk(Path(args.output_dir) / "trained_nel")
    
    

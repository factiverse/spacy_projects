import csv
import os
from pathlib import Path

import spacy
import typer
from spacy.kb import KnowledgeBase


def main(entities_loc: Path, vectors_model: str, kb_loc: Path, nlp_dir: Path):
    """Step 1: create the Knowledge Base in spaCy and write it to file"""

    # First: create a simpel model from a model with an NER component
    # To ensure we get the correct entities for this demo, add a simple entity_ruler as well.
    nlp = spacy.load(vectors_model, exclude="parser, tagger, lemmatizer")
    ruler = nlp.add_pipe("entity_ruler", after="ner")
    patterns = [{"label": "PERSON", "pattern": [{"LOWER": "emerson"}]}]
    ruler.add_patterns(patterns)
    nlp.add_pipe("sentencizer", first=True)

    name_dict, desc_dict = _load_entities(entities_loc)

    if "vectors" in nlp.meta and nlp.vocab.vectors.size:
        input_dim = nlp.vocab.vectors_length
    else:
        # raise ValueError(
        #     "The `nlp` object should have access to pretrained word vectors, "
        #     " cf. https://spacy.io/usage/models#languages."
        # )
        input_dim = len(nlp("Some text")._.trf_data.tensors[-1][0])

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=input_dim)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        # if "vectors" in nlp.meta:
        #     print("tok2vec")
        #     desc_enc = desc_doc.vector
        # else:
        # print("Transformer vector", desc_doc._.trf_data)
        desc_enc = desc_doc._.trf_data.tensors[-1][0]
        # print(desc, desc_enc)
        # Set arbitrary value for frequency
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)

    for qid, name in name_dict.items():
        # set 100% prior probability P(entity|alias) for each unique name
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])

    qids = name_dict.keys()
    probs = [0.3 for qid in qids]
    # ensure that sum([probs]) <= 1 when setting aliases
    kb.add_alias(alias="Emerson", entities=qids, probabilities=probs)  #

    print(f"Entities in the KB: {kb.get_entity_strings()}")
    print(f"Aliases in the KB: {kb.get_alias_strings()}")
    print()
    kb.to_disk(kb_loc)
    if not os.path.exists(nlp_dir):
        os.mkdir(nlp_dir)
    nlp.to_disk(nlp_dir)


def _load_entities(entities_loc: Path):
    """Helper function to read in the pre-defined entities we want to disambiguate to."""
    names = dict()
    descriptions = dict()
    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions


if __name__ == "__main__":
    typer.run(main)

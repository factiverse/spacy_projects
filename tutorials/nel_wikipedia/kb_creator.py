import logging
import sys

from spacy.kb import KnowledgeBase
from spacy.vocab import Vocab

import wiki_io as io

# logging.basicConfig(filename="spacy3_kb.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


import argparse
from pathlib import Path

import spacy
from tqdm import tqdm

spacy.require_gpu()

TESTING = False
from wiki_io import (
    ENTITY_ALIAS_PATH,
    ENTITY_DEFS_PATH,
    ENTITY_DESCR_PATH,
    ENTITY_FREQ_PATH,
    KB_FILE,
    KB_MODEL_DIR,
    LOG_FORMAT,
    PRIOR_PROB_PATH,
    TRAINING_DATA_FILE,
)


def create_kb(
    nlp,
    max_entities_per_alias,
    min_entity_freq,
    min_occ,
    entity_def_path,
    entity_descr_path,
    entity_alias_path,
    entity_freq_path,
    prior_prob_path,
    entity_vector_length,
):
    # Create the knowledge base from Wikidata entries
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=entity_vector_length)
    entity_list, filtered_title_to_id = _define_entities(
        nlp,
        kb,
        entity_def_path,
        entity_descr_path,
        min_entity_freq,
        entity_freq_path,
        entity_vector_length,
    )
    _define_aliases(
        kb,
        entity_alias_path,
        entity_list,
        filtered_title_to_id,
        max_entities_per_alias,
        min_occ,
        prior_prob_path,
    )
    return kb


def _define_entities(
    nlp,
    kb,
    entity_def_path,
    entity_descr_path,
    min_entity_freq,
    entity_freq_path,
    entity_vector_length,
):
    # read the mappings from file
    logger.info("Reading mappings from {}".format(entity_descr_path))
    title_to_id = io.read_title_to_id(entity_def_path)
    logger.info("Reading desc from {} ".format(entity_descr_path))
    id_to_descr = io.read_id_to_descr(entity_descr_path)

    logger.info(
        "Filtering entities with fewer than {} mentions".format(min_entity_freq)
    )
    entity_frequencies = io.read_entity_to_count(entity_freq_path)
    # filter the entities for in the KB by frequency, because there's just too much data (8M entities) otherwise
    (
        filtered_title_to_id,
        entity_list,
        description_list,
        frequency_list,
    ) = get_filtered_entities(
        title_to_id, id_to_descr, entity_frequencies, min_entity_freq
    )
    logger.info(
        "Kept {} entities from the set of {}".format(
            len(description_list), len(title_to_id.keys())
        )
    )

    logger.info("Getting entity embeddings")
    if TESTING:
        description_list = description_list[:1000]
        entity_list = entity_list[:1000]
        frequency_list = frequency_list[:1000]
    # embeddings = [nlp(desc)._.trf_data.tensors[-1][0] for desc in tqdm(description_list)]
    # check the length of the nlp vectors
    if "vectors" in nlp.meta and nlp.vocab.vectors.size:
        input_dim = nlp.vocab.vectors_length
        logger.info("Loaded pretrained vectors of size %s" % input_dim)
        embeddings = [nlp(desc).vector for desc in tqdm(description_list)]
    else:
        # raise ValueError(
        #     "The `nlp` object should have access to pretrained word vectors, "
        #     " cf. https://spacy.io/usage/models#languages."
        # )
        input_dim = len(nlp("Some text")._.trf_data.tensors[-1][0])
        logger.info("Creating transformer tensors of size %s" % input_dim)
        embeddings = []
        for i, desc in enumerate(tqdm(description_list)):
            try:
                embeddings.append(nlp(desc)._.trf_data.tensors[-1][0])
            except Exception as e:
                entity_list.pop(i)
                frequency_list.pop(i)
                logger.error(f"Expception processning {e}")
        # embeddings = [
        # nlp(desc)._.trf_data.tensors[-1][0] for desc in tqdm(description_list)
        # ]
    logger.info("Adding {} entities".format(len(entity_list)))
    kb.set_entities(
        entity_list=entity_list, freq_list=frequency_list, vector_list=embeddings
    )
    return entity_list, filtered_title_to_id


def _define_aliases(
    kb,
    entity_alias_path,
    entity_list,
    filtered_title_to_id,
    max_entities_per_alias,
    min_occ,
    prior_prob_path,
):
    logger.info("Adding aliases from Wikipedia and Wikidata")
    _add_aliases(
        kb,
        entity_list=entity_list,
        title_to_id=filtered_title_to_id,
        max_entities_per_alias=max_entities_per_alias,
        min_occ=min_occ,
        prior_prob_path=prior_prob_path,
    )


def get_filtered_entities(
    title_to_id, id_to_descr, entity_frequencies, min_entity_freq: int = 10
):
    filtered_title_to_id = dict()
    entity_list = []
    description_list = []
    frequency_list = []
    for title, entity in title_to_id.items():
        freq = entity_frequencies.get(title, 0)
        desc = id_to_descr.get(entity, None)
        if desc and freq > min_entity_freq:
            entity_list.append(entity)
            description_list.append(desc)
            frequency_list.append(freq)
            filtered_title_to_id[title] = entity
    return filtered_title_to_id, entity_list, description_list, frequency_list


def _add_aliases(
    kb, entity_list, title_to_id, max_entities_per_alias, min_occ, prior_prob_path
):
    wp_titles = title_to_id.keys()

    # adding aliases with prior probabilities
    # we can read this file sequentially, it's sorted by alias, and then by count
    logger.info("Adding WP aliases")
    with prior_prob_path.open("r", encoding="utf8") as prior_file:
        # skip header
        prior_file.readline()
        line = prior_file.readline()
        previous_alias = None
        total_count = 0
        counts = []
        entities = []
        while line:
            splits = line.replace("\n", "").split(sep="|")
            new_alias = splits[0]
            count = int(splits[1])
            entity = splits[2]
            if len(entities) >= 1000 and TESTING:
                break
            if new_alias != previous_alias and previous_alias:
                # done reading the previous alias --> output
                if len(entities) > 0:
                    selected_entities = []
                    prior_probs = []
                    for ent_count, ent_string in zip(counts, entities):
                        if ent_string in wp_titles:
                            wd_id = title_to_id[ent_string]
                            p_entity_givenalias = ent_count / total_count
                            selected_entities.append(wd_id)
                            prior_probs.append(p_entity_givenalias)

                    if selected_entities:
                        try:
                            kb.add_alias(
                                alias=previous_alias,
                                entities=selected_entities,
                                probabilities=prior_probs,
                            )
                        except ValueError as e:
                            logger.error(e)
                total_count = 0
                counts = []
                entities = []

            total_count += count

            if len(entities) < max_entities_per_alias and count >= min_occ:
                counts.append(count)
                entities.append(entity)
            previous_alias = new_alias

            line = prior_file.readline()


def main(
    output_dir,
    model,
    max_per_alias=20,
    min_freq=10,
    min_pair=5,
    loc_prior_prob=None,
    loc_entity_defs=None,
    loc_entity_alias=None,
    loc_entity_desc=None,
    loc_freq_path=None,
):
    entity_defs_path = (
        loc_entity_defs if loc_entity_defs else output_dir / ENTITY_DEFS_PATH
    )
    entity_alias_path = (
        loc_entity_alias if loc_entity_alias else output_dir / ENTITY_ALIAS_PATH
    )
    entity_descr_path = (
        loc_entity_desc if loc_entity_desc else output_dir / ENTITY_DESCR_PATH
    )
    entity_freq_path = loc_freq_path if loc_freq_path else output_dir / ENTITY_FREQ_PATH
    prior_prob_path = loc_prior_prob if loc_prior_prob else output_dir / PRIOR_PROB_PATH
    training_entities_path = output_dir / TRAINING_DATA_FILE
    kb_path = output_dir / KB_FILE

    logger.info("Creating KB with Wikipedia and WikiData")

    # STEP 0: set up IO
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # STEP 1: Load the NLP object
    logger.info("STEP 1: Loading NLP model {}".format(model))
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
    nlp = spacy.load(model, config=custom_config)
    if "vectors" in nlp.meta and nlp.vocab.vectors.size:
        input_dim = nlp.vocab.vectors_length
        logger.info("Will use pretrained vectors of size %s" % input_dim)
    else:
        # raise ValueError(
        #     "The `nlp` object should have access to pretrained word vectors, "
        #     " cf. https://spacy.io/usage/models#languages."
        # )
        input_dim = len(nlp("Some text")._.trf_data.tensors[-1][0])
        logger.info("Creating transformer tensors of size %s" % input_dim)
    # # STEP 2: create prior probabilities from WP
    if not prior_prob_path.exists():
        # It takes about 2h to process 1000M lines of Wikipedia XML dump
        logger.info("STEP 2: Writing prior probabilities to {}".format(prior_prob_path))
        if limit_prior is not None:
            logger.warning(
                "Warning: reading only {} lines of Wikipedia dump".format(limit_prior)
            )
        wp.read_prior_probs(wp_xml, prior_prob_path, limit=limit_prior)
    else:
        logger.info(
            "STEP 2: Reading prior probabilities from {}".format(prior_prob_path)
        )

    # STEP 3: calculate entity frequencies
    if not entity_freq_path.exists():
        logger.info(
            "STEP 3: Calculating and writing entity frequencies to {}".format(
                entity_freq_path
            )
        )
        io.write_entity_to_count(prior_prob_path, entity_freq_path)
    else:
        logger.info(
            "STEP 3: Reading entity frequencies from {}".format(entity_freq_path)
        )

    # STEP 4: reading definitions and (possibly) descriptions from WikiData or from file
    # if (not entity_defs_path.exists()) or (not descr_from_wp and not entity_descr_path.exists()):
    #     # It takes about 10h to process 55M lines of Wikidata JSON dump
    #     logger.info("STEP 4: Parsing and writing Wikidata entity definitions to {}".format(entity_defs_path))
    #     if limit_wd is not None:
    #         logger.warning("Warning: reading only {} lines of Wikidata dump".format(limit_wd))
    #     title_to_id, id_to_descr, id_to_alias = wd.read_wikidata_entities_json(
    #         wd_json,
    #         limit_wd,
    #         to_print=False,
    #         lang=lang,
    #         parse_descr=(not descr_from_wp),
    #     )
    #     io.write_title_to_id(entity_defs_path, title_to_id)

    #     logger.info("STEP 4b: Writing Wikidata entity aliases to {}".format(entity_alias_path))
    #     io.write_id_to_alias(entity_alias_path, id_to_alias)

    #     if not descr_from_wp:
    #         logger.info("STEP 4c: Writing Wikidata entity descriptions to {}".format(entity_descr_path))
    #         io.write_id_to_descr(entity_descr_path, id_to_descr)
    # else:
    #     logger.info("STEP 4: Reading entity definitions from {}".format(entity_defs_path))
    #     logger.info("STEP 4b: Reading entity aliases from {}".format(entity_alias_path))
    #     if not descr_from_wp:
    #         logger.info("STEP 4c: Reading entity descriptions from {}".format(entity_descr_path))

    # # STEP 5: Getting gold entities from Wikipedia
    # if (not training_entities_path.exists()) or (descr_from_wp and not entity_descr_path.exists()):
    #     logger.info("STEP 5: Parsing and writing Wikipedia gold entities to {}".format(training_entities_path))
    #     if limit_train is not None:
    #         logger.warning("Warning: reading only {} lines of Wikipedia dump".format(limit_train))
    #     wp.create_training_and_desc(wp_xml, entity_defs_path, entity_descr_path,
    #                                 training_entities_path, descr_from_wp, limit_train)
    #     if descr_from_wp:
    #         logger.info("STEP 5b: Parsing and writing Wikipedia descriptions to {}".format(entity_descr_path))
    # else:
    #     logger.info("STEP 5: Reading gold entities from {}".format(training_entities_path))
    #     if descr_from_wp:
    #         logger.info("STEP 5b: Reading entity descriptions from {}".format(entity_descr_path))

    # STEP 6: creating the actual KB
    # It takes ca. 30 minutes to pretrain the entity embeddings
    if not kb_path.exists():
        logger.info("STEP 6: Creating the KB at {}".format(kb_path))
        kb = create_kb(
            nlp=nlp,
            max_entities_per_alias=max_per_alias,
            min_entity_freq=min_freq,
            min_occ=min_pair,
            entity_def_path=entity_defs_path,
            entity_descr_path=entity_descr_path,
            entity_alias_path=entity_alias_path,
            entity_freq_path=entity_freq_path,
            prior_prob_path=prior_prob_path,
            entity_vector_length=input_dim,
        )
        kb.to_disk(kb_path)
        logger.info("kb entities: {}".format(kb.get_size_entities()))
        logger.info("kb aliases: {}".format(kb.get_size_aliases()))
        nlp.to_disk(output_dir / KB_MODEL_DIR)
    else:
        logger.info("STEP 6: KB already exists at {}".format(kb_path))

    logger.info("Done!")


def parse_args():
    parser = argparse.ArgumentParser("Create Spacy KB")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
        default="/local/home/vsetty/spacy_nel/data/spacy3_wikidata_kb",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="nb_core_news_lg for norwegian and en_core_web_lg for English",
        default="nb_core_news_lg",
    )
    parser.add_argument(
        "--kb_dir",
        type=str,
        required=True,
        help="Director containing KB",
        default="/local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_en_kb_train_output_230922",
    )
    # parser.add_argument(
    #     "--vector_len",
    #     type=int,
    #     required=True,
    #     help="Dimension of the vector",
    #     default=300,
    # )
    parser.add_argument(
        "--freq",
        type=int,
        required=True,
        help="Minimum number of frequency of entities to include",
        default=1,
    )
    return parser.parse_args()


#     # @plac.annotations(
#     # wd_json=("Path to the downloaded WikiData JSON dump.", "positional", None, Path),
#     # wp_xml=("Path to the downloaded Wikipedia XML dump.", "positional", None, Path),
#     output_dir=("Output directory", "positional", None, Path),
#     model=("Model name or path, should include pretrained vectors.", "positional", None, str),
#     max_per_alias=("Max. # entities per alias (default 10)", "option", "a", int),
#     min_freq=("Min. count of an entity in the corpus (default 20)", "option", "f", int),
#     min_pair=("Min. count of entity-alias pairs (default 5)", "option", "c", int),
#     entity_vector_length=("Length of entity vectors (default 64)", "option", "v", int),
#     loc_prior_prob=("Location to file with prior probabilities", "option", "p", Path),
#     loc_entity_defs=("Location to file with entity definitions", "option", "d", Path),
#     loc_entity_desc=("Location to file with entity descriptions", "option", "s", Path),
#     descr_from_wp=("Flag for using descriptions from WP instead of WD (default False)", "flag", "wp"),
#     limit_prior=("Threshold to limit lines read from WP for prior probabilities", "option", "lp", int),
#     limit_train=("Threshold to limit lines read from WP for training set", "option", "lt", int),
#     limit_wd=("Threshold to limit lines read from WD", "option", "lw", int),
#     lang=("Optional language for which to get Wikidata titles. Defaults to 'en'", "option", "la", str),
# )


def read_kb(kb_path, nlp_path):
    vocab = Vocab().from_disk(nlp_path / "vocab")
    kb = KnowledgeBase(vocab=vocab, entity_vector_length=768)
    kb.from_disk(kb_path)
    return kb


if __name__ == "__main__":
    args = parse_args()
    print(args)
    wikidata_wikipedia_preprocessed_dir = args.kb_dir
    main(
        output_dir=Path(args.output_dir),
        model=args.model,
        max_per_alias=10,
        min_freq=args.freq,
        min_pair=5,
        loc_prior_prob=Path(wikidata_wikipedia_preprocessed_dir) / PRIOR_PROB_PATH,
        loc_entity_defs=Path(wikidata_wikipedia_preprocessed_dir) / ENTITY_DEFS_PATH,
        loc_entity_alias=Path(wikidata_wikipedia_preprocessed_dir) / ENTITY_ALIAS_PATH,
        loc_entity_desc=Path(wikidata_wikipedia_preprocessed_dir) / ENTITY_DESCR_PATH,
        loc_freq_path=Path(wikidata_wikipedia_preprocessed_dir) / ENTITY_FREQ_PATH,
    )
    kb = read_kb(Path(args.output_dir) / KB_FILE, Path(args.output_dir) / KB_MODEL_DIR)
    print(len(kb))
    print(kb.entity_vector_length)
    print(kb.get_size_aliases())
    if TESTING:
        print(kb.get_alias_strings())
    print(kb.get_alias_candidates("Nel"))
# python kb_creator.py --model /local/home/vsetty/spacy_nel/data/paraphrase-multilingual-MiniLM-L12-v2/model-last --output_dir /local/home/vsetty/spacy_nel/data/nb_bert_ner/trained_kb --kb_dir /local/home/vsetty/spacy_nel/data/spacy_nel_wikidata_wikipedia_no_kb_train_output_250922/ --freq 1

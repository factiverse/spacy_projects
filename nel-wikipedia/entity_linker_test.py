import argparse
import spacy

parser = argparse.ArgumentParser("Spacy entity linker example code")
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="Path to the folder where the trained model for entity linking is present",
    default="spacy_nel_wikidata_wikipedia_en_train_output_sample_100/nlp"
)
parser.add_argument(
    "--text",
    type=str,
    required=True,
    help="Text to microfact",
    default="The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep."
)
args = parser.parse_args()  
nlp = spacy.load(args.model_path)
doc = nlp(args.text)
for ent in doc.ents:
    print(ent.text, ent.label_, ent.kb_id_)

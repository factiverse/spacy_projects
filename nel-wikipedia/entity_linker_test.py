import argparse
import spacy

parser = argparse.ArgumentParser("Spacy entity linker example code")
parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    help="Path to the folder where the trained model for entity linking is present",
    default="/data/spacy_nel/spacy_nel_norwegian_t50k_e5_d100_p0.3/nlp"
)
args = parser.parse_args()  
nlp = spacy.load(args.model_path)
doc = nlp("Hydrogenaksjen Nel ASA stiger hele ti prosent på Oslo Børs etter at selskapet torsdag meldte om en ordre på levering av elektrolyserør for 11 millioner euro (110 millioner kroner).")
for ent in doc.ents:
    print(ent)
    print(ent.text, ent.label_, ent.kb_id_, ent.kb_id, ent.doc)

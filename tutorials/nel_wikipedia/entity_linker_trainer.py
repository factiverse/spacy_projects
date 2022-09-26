import json
from pathlib import Path


def load_gold_dataset(data_path: Path):
    dataset = []
    json_loc = data_path / "gold_entities.jsonl"
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            text = example["clean_text"]
            entities = example["entities"]
            for entity in entities:
                QID = entity["entity"]
                offset = (entity["start"], entity["end"])
                entity_label = entity["alias"]
                entities = [(offset[0], offset[1], entity_label)]
                links_dict = {QID: 1.0}
            dataset.append((text, {"links": {offset: links_dict}, "entities": entities}))
    return dataset

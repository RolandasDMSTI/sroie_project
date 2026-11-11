import os
import glob
import json
import re

from sroie_dataset import bbox_train_dir, annt_train_dir
from sroie_dataset import bbox_test_dir, annt_test_dir
from sroie_dataset import pattern, ENTITIES


def sanity_check(bbox_dir=bbox_train_dir, annt_dir=annt_train_dir):
    for file in glob.glob(pattern, root_dir=bbox_dir):
        with open(bbox_dir + file, encoding="cp1252") as f:
            for line in f:
                parts = line.strip().split(",", maxsplit=8) 
                if len(parts) < 9:
                    print(file, repr(line))
                    continue
                # print(parts, int(parts[7]) - int(parts[1]))
                if(parts[1] != parts[3] or parts[5] != parts[7]):
                    print(file, parts)
                if "  " in parts[8]:
                    print(file, "words", repr(parts[8]))                

        if not os.path.exists(annt_dir + file):
            print("Annotation missing", file)
            continue
        with open(annt_dir + file) as f:
            annt = json.load(f)
        for field in ENTITIES:
            try:
                if "  " in annt[field]:
                    print(file, field, repr(annt[field]))
            except Exception as ex:
                print(file, "missing", ex)


def check_file_pairing(bbox_dir=bbox_train_dir, annt_dir=annt_train_dir):
    for file in glob.glob("X???????????.txt", root_dir=bbox_dir):
        count = len(glob.glob(file, root_dir=annt_dir))
        if count != 1:
            print(f"{count} {file}")

# sanity_check()
# sanity_check(bbox_test_dir, annt_test_dir)

# check_file_pairing()


from sroie_dataset import read_ocr_text

def read_text(filename):
    words, _ = read_ocr_text(filename)
    return " ".join(words)

def compact_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())

# def fuzzy_match(what, where):
#     from difflib import SequenceMatcher


from sroie_dataset import find_ocr_match

def match(field_name, bbox_dir=bbox_train_dir, annt_dir=annt_train_dir, threshold=100):
    for file_name in glob.glob(pattern, root_dir=bbox_dir):
        text = read_text(bbox_dir + file_name)

        if not os.path.exists(annt_dir + file_name):
            print("Annotation missing", file_name)
            continue
        with open(annt_dir + file_name) as f:
            annt = json.load(f)
        
        if annt.get(field_name, "") == "":
            continue
        where = text.find(annt[field_name])
        if where < 0:
            best_start, best_end, best_score = find_ocr_match(annt[field_name], text)
            if best_score < threshold:
                print(file_name)
                print(f"{field_name}:", annt[field_name])
                print(text)
                print(best_start, best_end, best_score)
                print(text[best_start : best_end])
                print(annt[field_name])
                print("-" * 100)
            pass
        else:
            more = text.find(annt[field_name], where + 1)
            if more > -1:
                print("DUP at", more)
                print(annt[field_name])
                print(text)

# match("address")
# match("total")

from sroie_dataset import plain_sroie_dataset

dataset = plain_sroie_dataset()
for name, split in dataset.items():
    print(f"--- {name} ---")
    sizes = split["sizes"]
    print(f"{min(sizes) = }")
    print(f"{max(sizes) = }")
    import numpy as np
    sizes = np.concatenate(split["sizes"])
    print(f"{sum(sizes) / len(sizes) = }")
    print("percentiles:")
    print(np.percentile(sizes, [0, 1, 5, 25, 50, 75, 95, 99, 100]))

import os
import glob
import json
import warnings
from datasets import DatasetDict, Dataset
from rapidfuzz import fuzz

base_dir = "../"
bbox_train_dir = base_dir + "t1/0325updated.task1train(626p)/"
annt_train_dir = base_dir + "t2/0325updated.task2train(626p)/"
bbox_test_dir = base_dir + "t3/text/"
annt_test_dir = base_dir + "t3/annt/"
pattern = "X???????????.txt"

ENTITIES = ["company", "date", "address", "total"]
LABELS = ["O"] + [f"{p}-{f.upper()}" for f in ENTITIES for p in "BI"]
tag2id = {tag: i for i, tag in enumerate(LABELS)}
id2tag = {i: tag for tag, i in tag2id.items()}

def read_ocr_text(filename):
    with open(filename, encoding="cp1252") as f:
        words = []
        sizes = []
        for line in f:
            parts = line.strip().split(",", maxsplit=8)
            if len(parts) < 9:
                continue
            font_size = int(parts[7]) - int(parts[1])
            for word in parts[8].split():
                words.append(word)
                sizes.append(font_size)
    return words, sizes


def read_annotations(filename):
    if not os.path.exists(filename):
        warnings.warn(f"Annotation missing for {filename}", UserWarning)
        return None, None, None, None

    with open(filename) as f:
        annt = json.load(f)

    company = annt.get("company", "")
    date = annt.get("date", "")
    address = annt.get("address", "")
    total = annt.get("total", "")

    for field_name in ENTITIES:
        if annt.get(field_name, "") == "":
            warnings.warn(f"No annotation for {field_name} in {filename}", UserWarning)

    return company, date, address, total


def find_ocr_match(what: str, where: str):
    best_score = 0
    best_start = -1
    best_end = -1

    n = len(what)

    # OCR can insert/delete characters, so try windows somewhat
    # shorter and longer than `a`.
    min_len = max(1, int(n * 0.7))
    max_len = min(len(where), int(n * 1.3) + 1)

    for start in range(len(where)):
        for length in range(min_len, max_len + 1):
            end = start + length
            if end > len(where):
                break
            candidate = where[start:end]
            score = fuzz.ratio(what, candidate)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    return best_start, best_end, best_score


def tag_field(field_text, btag, words, ner_tags, *, last=False, log_fuzzy=False):
    # TODO improvement for TOTAL field:
    # find all candidates
    # mark the nearest to "TOTAL" text on the left
    # the last one otherwise
    if field_text == "":
        return ner_tags
    text = " ".join(words)
    where = text.find(field_text) if not last else text.rfind(field_text)
    if where > -1:
        start = text[:where].count(" ")
        word_gaps = field_text.count(" ")
        ner_tags[start:start + word_gaps + 1] = [btag] + [btag + 1] * word_gaps
        # more = text.find(field, where + 1) # and reverse logic as well
        # if more > -1:
        #     print(btag, more, repr(field))
        #     print(words)
    else:
        best_start, best_end, best_score = find_ocr_match(field_text, text)
        start = text[:best_start].count(" ") + (text[best_start] == " ") # skip leading space, if any
        end = text[:best_end].count(" ") - (text[best_end - 1] == " ") # skip trailing space, if any
        word_count = end - start + 1
        ner_tags[start:start + word_count] = [btag] + [btag + 1] * (word_count - 1)
        marked = " ".join(words[start:start + word_count])
        # if marked != text[best_start:best_end]: 
        #     print(f"Fuzzy match for {id2tag[btag][2:]}, marked != text[best_start:best_end]:")
        #     print(f"{best_start=}, {best_end=}, {best_score=}")
        #     print("searched =", repr(field_text))
        #     print("found =", repr(text[best_start:best_end]))
        #     print("marked =", repr(marked))
        #     print(words)
        if log_fuzzy:
            print(f"Fuzzy match for {id2tag[btag][2:]}:")
            print(f"{best_start=}, {best_end=}, {best_score=}")
            print("searched =", repr(field_text))
            print("found =", repr(text[best_start:best_end]))
            print("marked =", repr(marked))
            print(words)
    return ner_tags


def prepare_receipt_row(file, bbox_dir, annt_dir):
        words, sizes = read_ocr_text(bbox_dir + file)
        company, date, address, total = read_annotations(annt_dir + file)
        if company is None:
            return words, sizes, None, None, None, None, None
        ner_tags = [tag2id["O"]] * len(words)
        tag_field(company, tag2id["B-COMPANY"], words, ner_tags)
        tag_field(date, tag2id["B-DATE"], words, ner_tags)
        tag_field(address, tag2id["B-ADDRESS"], words, ner_tags)
        tag_field(total, tag2id["B-TOTAL"], words, ner_tags, last=True, log_fuzzy=True)
        return words, sizes, ner_tags, company, date, address, total

def validate_receipt_row(file, row):
    words, sizes, ner_tags, company, date, address, total = row
    assert len(words) == len(sizes)
    assert ner_tags is None or len(words) == len(ner_tags)
    if ner_tags is not None:
        for fieldname in ENTITIES:
            # TODO: Some field validation
            pass
    else:
        print("No annotations for the ", file)

def make_sroie_split(bbox_dir, annt_dir):
    rows = []
    for filename in glob.glob(pattern, root_dir=bbox_dir):
        row = prepare_receipt_row(filename, bbox_dir, annt_dir)
        validate_receipt_row(filename, row)
        words, sizes, ner_tags, company, date, address, total = row
        if ner_tags is not None:
            rows.append({
                "filename": filename,
                "tokens": words,
                "sizes": sizes,
                "ner_tags": ner_tags,
                "company": company,
                "date": date,
                "address": address,
                "total": total,
            })
    return Dataset.from_list(rows)

def plain_sroie_dataset():
    train = make_sroie_split(bbox_train_dir, annt_train_dir)
    test = make_sroie_split(bbox_test_dir, annt_test_dir)
    split = train.train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": test,
    })
    return dataset

# file = "X51005447832.txt"
# row = prepare_receipt_row(file, bbox_train_dir, annt_train_dir)
# file = "X51006008092.txt"
# row = prepare_receipt_row(file, bbox_test_dir, annt_test_dir)
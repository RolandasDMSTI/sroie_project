import torch

NEG_INF = -10000.0

def make_bio_constraints(tag2id):
    num_tags = len(tag2id)

    transition_mask = torch.zeros(num_tags, num_tags)
    start_mask = torch.zeros(num_tags)
    end_mask = torch.zeros(num_tags)

    for from_tag, from_id in tag2id.items():
        for to_tag, to_id in tag2id.items():
            if not allowed_transition(from_tag, to_tag):
                transition_mask[from_id, to_id] = NEG_INF

    # An I-X tag cannot start a sequence.
    for tag, tag_id in tag2id.items():
        if tag.startswith("I-"):
            start_mask[tag_id] = NEG_INF

    # Standard BIO allows every tag to end a sequence.
    pass 

    return transition_mask, start_mask, end_mask

def allowed_transition(from_tag, to_tag):
    if not to_tag.startswith("I-"):
        return True

    if from_tag == "O":
        return False

    if from_tag.startswith(("B-", "I-")):
        return from_tag[2:] == to_tag[2:]

    return False

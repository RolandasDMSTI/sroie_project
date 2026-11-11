import random
import numpy
import torch
from seqeval.metrics import precision_score, recall_score, f1_score

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, id2tag, device):

    def _to_float_metric(value):
        if isinstance(value, list):
            raise ValueError("list is not expected")
        return float(value)

    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            features = batch["features"].to(device)
            sizes = batch["sizes"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            predictions = model.decode(input_ids, mask, features, sizes, attention_mask)

            # Convert each sequence to tag strings
            for i, pred in enumerate(predictions):
                length = int(mask[i].sum().item())

                true_tags = tags[i][:length].tolist()
                pred_tags = pred[:length]

                all_true.append([
                    id2tag[tag_id]
                    for tag_id in true_tags
                ])

                all_pred.append([
                    id2tag[tag_id]
                    for tag_id in pred_tags
                ])

    precision = _to_float_metric(precision_score(all_true, all_pred))
    recall = _to_float_metric(recall_score(all_true, all_pred))
    f1 = _to_float_metric(f1_score(all_true, all_pred))

    return precision, recall, f1

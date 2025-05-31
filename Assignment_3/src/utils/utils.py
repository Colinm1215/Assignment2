import collections
from pathlib import Path
import json
from datasets import load_dataset


def load_squad_dataset(only_answerable: bool = True):
    ds = load_dataset("squad_v2")
    train, dev = ds["train"], ds["validation"]

    if only_answerable:
        train = filter_answerable(train)
        dev = filter_answerable(dev)

    return train, dev

def filter_answerable(dataset):
    return dataset.filter(lambda ex: len(ex["answers"]["text"]) > 0)

def add_context(examples, sample_mapping):
    return [examples["context"][i] for i in sample_mapping]

def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = examples["answers"][sample_idx]
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        seq_ids = tokenized_examples.sequence_ids(i)
        ctx_start = seq_ids.index(1)
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

        if not (offsets[ctx_start][0] <= start_char <= end_char <= offsets[ctx_end][1]):
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start = ctx_start
        while token_start <= ctx_end and offsets[token_start][0] <= start_char:
            token_start += 1
        token_start -= 1

        token_end = ctx_end
        while token_end >= ctx_start and offsets[token_end][1] >= end_char:
            token_end -= 1
        token_end += 1

        start_positions.append(token_start)
        end_positions.append(token_end)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def prepare_eval_features(examples,
                          tokenizer,
                          max_length,
                          doc_stride):
    tok = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tok.pop("overflow_to_sample_mapping")
    tok["example_id"] = [examples["id"][i] for i in sample_mapping]
    tok["context"]    = add_context(examples, sample_mapping)

    new_offsets = []
    for i, offsets in enumerate(tok["offset_mapping"]):
        seq_ids = tok.sequence_ids(i)
        new_offsets.append([
            (o if s == 1 else (0, 0)) for o, s in zip(offsets, seq_ids)
        ])
    tok["offset_mapping"] = new_offsets
    return tok


def save_predictions(pred_dict, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pred_dict, f, ensure_ascii=False, indent=2)
    return path

def save_dataset_squadFormat(dataset, path):
    data = []
    grouped = collections.defaultdict(list)
    for example in dataset:
        title = example.get("title", "default")
        grouped[title].append({
            "context": example["context"],
            "qas": [{
                "id": example["id"],
                "question": example["question"],
                "answers": [
                    {"text": text, "answer_start": start}
                    for text, start in zip(example["answers"]["text"], example["answers"]["answer_start"])
                ],
                "is_impossible": len(example["answers"]["text"]) == 0
            }]
        })
    for title, paragraphs in grouped.items():
        data.append({"title": title, "paragraphs": paragraphs})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": "v2.0", "data": data}, f, indent=2)

def run_squad_evaluator(dev_json, pred_json, out_path=None, na_prob_path=None):
    from pathlib import Path
    import subprocess

    dev_json, pred_json = map(Path, (dev_json, pred_json))
    if out_path is None:
        out_path = pred_json.with_name("eval.json")

    eval_script = Path(__file__).resolve().parent.parent / "tests" / "evaluation.py"

    if not eval_script.exists():
        print("Evaluation script not found")
        return None

    print("\nRunning evaluation …\n")

    cmd = [
        "python", str(eval_script),
        str(dev_json), str(pred_json),
        "--out-file", str(out_path)
    ]

    if na_prob_path is not None:
        cmd.extend(["--na-prob-file", str(na_prob_path)])

    subprocess.run(cmd, check=True)
    return out_path

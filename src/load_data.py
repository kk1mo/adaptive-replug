import random
from typing import List, Tuple

from datasets import load_dataset, Dataset


def load_train_data(train_url, docs_url):
    trainset = load_dataset(train_url, split="train")
    corpus_texts = load_dataset(docs_url, split="train")
    return trainset, corpus_texts


def load_mmlu():
    categories = {
        "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other": ["other", "business", "health"],
    }

    def get_category(subject):
        subject_lower = subject.lower()
        for category, keywords in categories.items():
            if any(kw in subject_lower for kw in keywords):
                return category
        return "other"

    mmlu_dev  = load_dataset("cais/mmlu", "all", split="dev",  trust_remote_code=True)
    mmlu_test = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)

    mmlu_dev  = mmlu_dev.map(lambda ex: {"category": get_category(ex["subject"])})
    mmlu_test = mmlu_test.map(lambda ex: {"category": get_category(ex["subject"])})

    n_subjects = len(set(mmlu_test["subject"]))
    category_counts = {cat: mmlu_test["category"].count(cat) for cat in categories}
    print(f"dev={len(mmlu_dev):,}  test={len(mmlu_test):,}  subjects={n_subjects}")
    print(f"category distribution (test): { {k: v for k, v in category_counts.items()} }")
    return mmlu_dev, mmlu_test
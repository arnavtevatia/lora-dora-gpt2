# Data

We use WikiText-2 (wikitext-2-raw-v1), a standard language modeling benchmark of curated Wikipedia articles.

The dataset is publicly available via HuggingFace Datasets and does not need to be downloaded manually. The training script fetches it automatically:

    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

Dataset stats:
- Train: ~2.1M tokens
- Validation: ~217K tokens
- Test: ~245K tokens

No authentication or special access required.
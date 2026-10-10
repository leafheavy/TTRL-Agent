"""Validate dataset rows on CPU before initializing training/model workers."""


def validate_dataset(dataset, split):
    """Exercise the real __getitem__ path, including tool schemas and tokenization."""
    summary = {"samples": len(dataset), "max_prompt_tokens": 0, "longest_index": None}
    for index in range(len(dataset)):
        try:
            row = dataset[index]
        except Exception as exc:
            raise ValueError(
                f"{split} data preflight failed at dataset index {index}, before model initialization: {exc}"
            ) from exc
        if "raw_prompt_ids" in row:
            length = len(row["raw_prompt_ids"])
        elif "attention_mask" in row:
            mask = row["attention_mask"]
            length = int(mask.sum()) if hasattr(mask, "sum") else sum(mask)
        else:
            length = len(row["input_ids"])
        if length > summary["max_prompt_tokens"]:
            summary["max_prompt_tokens"] = length
            summary["longest_index"] = row.get("index", index)
    print(f"[data_preflight] {split}: {summary}")
    return summary


def run_data_preflight(config, train_dataset, val_dataset):
    """Check all BFCL rows by default; return True for a requested CPU-only run."""
    preflight_only = config.get("trainer", {}).get("preflight_only", False)
    ttrl = config.get("ttrl", {})
    bfcl = config.get("bfcl_supervised", {}).get("enabled", False) or (
        ttrl.get("enable", False) and ttrl.get("reward_mode", "math") == "bfcl"
    )
    if bfcl or preflight_only:
        validate_dataset(train_dataset, "train")
        validate_dataset(val_dataset, "validation")
    if preflight_only:
        print("[data_preflight] Complete. Model workers were not initialized (trainer.preflight_only=True).")
    return bool(preflight_only)

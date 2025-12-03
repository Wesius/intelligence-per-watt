
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np

# Output Configuration
OUTPUT_DIR = Path("intelligence-per-watt/src/ipw/datasets/ipw_pro/data/mixed_10k_seed42")
SEED = 42
SAMPLES_PER_DATASET = 2500

def format_options(options: List[str]) -> str:
    """Helper to format multiple choice options."""
    rendered = []
    for idx, option in enumerate(options):
        letter = chr(ord("A") + idx)
        rendered.append(f"{letter}. {option}")
    return "\n".join(rendered)

def process_wildchat(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Source: allenai/WildChat
    Extract first User/Assistant exchange.
    """
    conversation = sample.get("conversation")
    if not conversation:
        return None

    # Find first user message and subsequent assistant message
    problem = None
    answer = None
    
    for i in range(len(conversation) - 1):
        msg = conversation[i]
        next_msg = conversation[i+1]
        
        if msg.get("role") == "user" and next_msg.get("role") == "assistant":
            problem = msg.get("content")
            answer = next_msg.get("content")
            break
            
    if not problem or not answer:
        return None

    return {
        "problem": problem,
        "answer": answer,
        "subject": "general",
        "dataset_metadata": json.dumps({
            "config": {
                "dataset_name": "allenai/WildChat",
                "verification_method": "wildchat"
            },
            "original_id": sample.get("conversation_hash"),
            "model": sample.get("model")
        })
    }

def process_natural_reasoning(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Source: facebook/natural_reasoning
    """
    problem = sample.get("question")
    answer = sample.get("reference_answer")
    
    if not problem or not answer:
        return None

    return {
        "problem": problem,
        "answer": answer,
        "subject": "reasoning",
        "dataset_metadata": json.dumps({
            "config": {
                "dataset_name": "facebook/natural_reasoning",
                "verification_method": "natural_reasoning"
            },
            "source": sample.get("source")
        })
    }

def process_mmlu_pro(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Source: TIGER-Lab/MMLU-Pro
    """
    question = str(sample.get("question") or "").strip()
    options = [str(option) for option in sample.get("options", []) or []]
    answer = str(sample.get("answer") or "").strip().upper()
    subject = str(sample.get("category") or "general").strip()

    if not question or not answer:
        return None

    prompt_parts = [question]
    if options:
        prompt_parts.append("")
        prompt_parts.append("Options:")
        prompt_parts.append(format_options(options))
    prompt_parts.append("")
    prompt_parts.append("Respond with the correct letter.")
    
    problem = "\n".join(prompt_parts)

    return {
        "problem": problem,
        "answer": answer,
        "subject": subject,
        "dataset_metadata": json.dumps({
            "config": {
                "dataset_name": "TIGER-Lab/MMLU-Pro",
                "verification_method": "mmlu-pro"
            },
            "question_id": sample.get("question_id"),
            "options": options
        })
    }

def process_supergpqa(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Source: m-a-p/SuperGPQA
    """
    question = str(sample.get("question") or "").strip()
    options_raw = sample.get("options") or []
    options = [str(option).strip() for option in options_raw if str(option).strip()]
    answer_letter = str(sample.get("answer_letter") or "").strip().upper()
    
    subject = str(
        sample.get("subfield")
        or sample.get("field")
        or sample.get("discipline")
        or "general"
    ).strip()

    if not question or not options or not answer_letter:
        return None

    prompt_sections = [question, "", "Options:", format_options(options), "", "Respond with the correct letter only."]
    problem = "\n".join(section for section in prompt_sections if section).strip()

    return {
        "problem": problem,
        "answer": answer_letter,
        "subject": subject,
        "dataset_metadata": json.dumps({
            "config": {
                "dataset_name": "m-a-p/SuperGPQA",
                "verification_method": "supergpqa"
            },
            "uuid": sample.get("uuid"),
            "field": sample.get("field")
        })
    }

def get_random_samples(dataset_name, split, process_fn, num_samples, seed):
    print(f"Loading {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return []

    # Shuffle and select a buffer (2x needed to be safe against invalid rows)
    print(f"Shuffling and selecting candidates from {dataset_name}...")
    shuffled = ds.shuffle(seed=seed)
    
    # We might need more than num_samples if some are invalid
    # Take a larger slice first
    candidates = shuffled.select(range(min(len(shuffled), num_samples * 2)))
    
    valid_records = []
    for row in candidates:
        processed = process_fn(row)
        if processed:
            valid_records.append(processed)
            if len(valid_records) >= num_samples:
                break
    
    print(f"Collected {len(valid_records)} valid samples from {dataset_name}")
    return valid_records

def main():
    all_records = []

    # 1. WildChat
    wildchat_records = get_random_samples(
        "allenai/WildChat", 
        "train", 
        process_wildchat, 
        SAMPLES_PER_DATASET, 
        SEED
    )
    all_records.extend(wildchat_records)

    # 2. NaturalReasoning
    nr_records = get_random_samples(
        "facebook/natural_reasoning", 
        "train", 
        process_natural_reasoning, 
        SAMPLES_PER_DATASET, 
        SEED
    )
    all_records.extend(nr_records)

    # 3. MMLU-Pro
    mmlu_records = get_random_samples(
        "TIGER-Lab/MMLU-Pro", 
        "test", 
        process_mmlu_pro, 
        SAMPLES_PER_DATASET, 
        SEED
    )
    all_records.extend(mmlu_records)

    # 4. SuperGPQA
    gpqa_records = get_random_samples(
        "m-a-p/SuperGPQA", 
        "train", 
        process_supergpqa, 
        SAMPLES_PER_DATASET, 
        SEED
    )
    all_records.extend(gpqa_records)

    print(f"Total records collected: {len(all_records)}")

    # Create HF Dataset
    final_dataset = Dataset.from_list(all_records)
    
    # Shuffle the mix
    final_dataset = final_dataset.shuffle(seed=SEED)

    # Save
    print(f"Saving to {OUTPUT_DIR}...")
    final_dataset.save_to_disk(str(OUTPUT_DIR))
    print("Done!")

if __name__ == "__main__":
    main()

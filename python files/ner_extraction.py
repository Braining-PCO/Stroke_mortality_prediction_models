# Import Libraries
import re
import pandas as pd
from typing import List
from transformers import pipeline
from openai import AsyncOpenAI
import os
import asyncio
import nest_asyncio

nest_asyncio.apply()


# Text Cleaning
def preprocess_medical_note_v2(note):
    """
    Clean medical notes by removing annotations, excessive whitespace,
    newline breaks, and special characters, returning normalized text.
    """
    if pd.isna(note):
        return ""
    note_cleaned = re.sub(r'\[\*\*(.*?)\*\*\]', r'\1', note)
    note_cleaned = re.sub(r'\[(.*?)\]', r'\1', note_cleaned)
    note_cleaned = re.sub(r'\s{2,}', ' ', note_cleaned)
    note_cleaned = re.sub(r'\n+', ' ', note_cleaned)
    note_cleaned = re.sub(r'\s*:\s*', ': ', note_cleaned)
    note_cleaned = re.sub(r"\*{1,2}", "", note_cleaned)
    return note_cleaned.strip()


# Helper Functions
def tokenize(text: str) -> List[str]:
    """
    Split text into tokens consisting of words and punctuation.
    """
    return re.findall(r"\w+|[^\w\s]", text)


def extract_contexts(text: str,
                     entities: List[str],
                     window: int = 5) -> List[dict]:
    """
    Locate entities in text and extract surrounding context windows
    for downstream relevance filtering.
    """
    tokens = tokenize(text)
    results = []
    for entity in entities:
        phrase_tokens = tokenize(entity)
        length = len(phrase_tokens)
        for i in range(len(tokens) - length + 1):
            if tokens[i:i + length] == phrase_tokens:
                start = max(0, i - window)
                end = min(len(tokens), i + length + window)
                context = " ".join(tokens[start:end])
                results.append({
                    "word": entity,
                    "context": context,
                    "index": i
                })
    return results


def merge_subwords(ner_output):
    """
    Merge WordPiece-style NER subword tokens into complete words and
    return aggregated entity outputs.
    """
    merged = []
    buffer = {"word": "", "entity_group": None}

    for e in ner_output:
        if e["word"].startswith("##"):
            buffer["word"] += e["word"][2:]
        else:
            if buffer["word"]:
                merged.append(buffer.copy())
            buffer = {"word": e["word"], "entity_group": e["entity_group"]}

    merged.append(buffer)
    return merged


# Entity Filters
entity_groups_to_remove = {
    'NONBIOLOGICAL_LOCATION', 'DATE', 'HISTORY', 'TIME', 'AREA',
    'FAMILY_HISTORY', 'VOLUME', 'SEX', 'AGE', 'COREFERENCE', 'ADMINISTRATION',
    'DISTANCE'
}

important_groups = {
    'THERAPEUTIC_PROCEDURE', 'SIGN_SYMPTOM', 'DIAGNOSTIC_PROCEDURE',
    'MEDICATION', 'LAB_VALUE', 'DISEASE_DISORDER', 'BIOLOGICAL_STRUCTURE',
    'SEVERITY', 'DOSAGE', 'AGE', 'SEX'
}


def filter_labels(entities):
    """
    Keep only entities belonging to medically important semantic groups.
    """
    return [e for e in entities if e['entity_group'] in important_groups]


# Async Entity Extraction (LLM relevance check only)
async def extract_medical_entities_async(note, ner_model, client=None):
    """
    Extract medical entities from a note using a NER model, optionally
    verifying relevance with an LLM. Returns entities and token/cost stats.
    """
    if not isinstance(note, str) or note.strip() == "":
        return {
            "entities": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        }

    processed_text = preprocess_medical_note_v2(note)
    raw_entities = ner_model(processed_text)
    raw_merge_subwords = merge_subwords(raw_entities)
    filtered_entities = filter_labels(raw_merge_subwords)

    if not filtered_entities:
        return {
            "entities": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        }

    entity_words = [e['word'] for e in filtered_entities if 'word' in e]
    entities_with_context = extract_contexts(processed_text,
                                             entity_words,
                                             window=5)

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    relevant_entities = []

    price_per_1k_input = 0.00015
    price_per_1k_output = 0.0006

    if client:
        for entry in entities_with_context:
            word = entry["word"]
            context = entry["context"]
            prompt = f"""
Does the term '{word}' in the context below refer to a medically relevant concept?  
    A medically relevant concept includes:
    - Diseases, symptoms, or injuries
    - Anatomical structures (brain, muscles, organs)
    - Functional abilities or impairments (gait, muscle strength)
    - Treatments or therapies (OT, TF, rehabilitation)
    - Diagnostic tests, codes, or classifications (ICF, ICD)
    - Severity of the condition (severe, mild, low,..)
Context:
---
{context}
---
Answer only "YES" or "NO". Be generous if it's potentially related to healthcare or rehabilitation.
"""

            try:
                response = await client.responses.create(model="gpt-4o",
                                                         input=prompt.strip())
                output_text = response.output_text.strip().upper()

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                cost = (input_tokens / 1000 * price_per_1k_input) + (
                    output_tokens / 1000 * price_per_1k_output)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += cost

                if output_text == "YES":
                    relevant_entities.append(entry["word"])

            except Exception as e:
                print(f"Skipping LLM call for '{word}': {e}")

    else:
        relevant_entities = [e['word'] for e in filtered_entities]

    final_entities = [
        f"{e['word']}: {e['entity_group']}" for e in filtered_entities
        if e['word'] in relevant_entities
    ]

    return {
        "entities": final_entities,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost": total_cost,
    }


# Execute Pipeline
def run_pipeline(dataset_path,
                 save_path,
                 text_column="text",
                 batch_size=100,
                 limit=None,
                 use_llm=False):
    """
    Load the dataset, run medical NER (and optional LLM relevance filtering)
    in batches, and save annotated output with cost tracking.
    """

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    df = pd.read_csv(dataset_path)
    if limit:
        df = df.head(limit)

    ner_model = pipeline("token-classification",
                         model="blaze999/Medical-NER",
                         aggregation_strategy='simple')

    client = None
    if use_llm:
        client = AsyncOpenAI(api_key="OPENAI_API_KEY")

    if os.path.exists(save_path):
        processed_df = pd.read_csv(save_path)
        start_index = len(processed_df)
        print(f"Resuming from row {start_index}")
    else:
        processed_df = pd.DataFrame(columns=list(df.columns) + ["entities"])
        start_index = 0

    print(f"Starting batch processing from {start_index} / {len(df)}")

    async def process_batch(batch):
        """
        Asynchronously process a batch of notes for entity extraction.
        """
        tasks = [
            extract_medical_entities_async(note, ner_model, client)
            for note in batch[text_column]
        ]
        return await asyncio.gather(*tasks)

    for i in range(start_index, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        print(f"Processing batch {i} → {i + len(batch)}")

        if use_llm:
            results = asyncio.run(process_batch(batch))
            entities_col = []
            batch_input_tokens, batch_output_tokens, batch_cost = 0, 0, 0

            for res in results:
                entities_col.append(res["entities"])
                batch_input_tokens += res["input_tokens"]
                batch_output_tokens += res["output_tokens"]
                batch_cost += res["cost"]

            total_input_tokens += batch_input_tokens
            total_output_tokens += batch_output_tokens
            total_cost_usd += batch_cost
            print(
                f"Batch cost: ${batch_cost:.4f} | Total so far: ${total_cost_usd:.4f}"
            )
        else:
            entities_col = [
                extract_medical_entities_async(note, ner_model)["entities"]
                for note in batch[text_column]
            ]

        batch = df.iloc[i:i + batch_size].copy()
        batch["entities"] = entities_col
        processed_df = pd.concat([processed_df, batch], ignore_index=True)
        processed_df.to_csv(save_path, index=False)
        print(f"Saved progress → {save_path}")

    print(f"All batches completed.\n Total cost: ${total_cost_usd:.4f}")
    return processed_df


# Run
augmented_df = run_pipeline(
    dataset_path="INPUT_FILE_PATH_HERE/medical_notes.csv",
    text_column="text",
    limit=100,
    use_llm=True,
    save_path="FILE_PATH_HERE/medical_entities.csv")

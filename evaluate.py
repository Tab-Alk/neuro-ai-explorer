import os
import json
from dotenv import load_dotenv
import pandas as pd
import sys
import time
import logging

# --- Setup Logging for Cleaner Output ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment and Path Setup ---
# Set TOKENIZERS_PARALLELISM to false before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_engine import query_rag
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

def load_test_set(filepath="evaluation/gold_standard_test_set.json"):
    logging.info("Loading test set...")
    with open(filepath, 'r') as f:
        test_set = json.load(f)
    logging.info(f"Loaded {len(test_set)} test cases.")
    return test_set

def run_rag_for_test_case(query, api_key):
    answer, sources = query_rag(query, api_key=api_key)
    retrieved_context = [doc.page_content for doc in sources]
    return answer, retrieved_context

def main():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logging.error("CRITICAL ERROR: GROQ_API_KEY not found in .env file.")
        return

    test_cases = load_test_set()
    rows = []
    
    logging.info("Processing test cases with the RAG system...")
    for i, test_case in enumerate(test_cases):
        logging.info(f"Processing case {i+1}/{len(test_cases)}: {test_case['question']}")
        try:
            generated_answer, retrieved_context = run_rag_for_test_case(test_case['question'], api_key=groq_key)
            rows.append({
                "question": test_case['question'],
                "answer": generated_answer,
                "contexts": retrieved_context,
                "ground_truth": test_case['ground_truth_answer'],
                "ground_truth_context": test_case['ground_truth_context']
            })
        except Exception as e:
            logging.error(f"ERROR processing case {i+1}: {e}")

    if not rows:
        logging.warning("No successful results to evaluate. Exiting.")
        return

    dataset = Dataset.from_list(rows)
    
    ragas_llm = ChatOpenAI(
        temperature=0,
        model_name="llama3-8b-8192",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=groq_key,
        timeout=300,
    )
    ragas_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- Chunked evaluation to stay under Groq's token-per‑minute limit ---
    CHUNK_SIZE = 5     # ≤ 5 questions per batch
    COOLDOWN   = 200   # wait 2 minutes between batches

    metrics_to_evaluate = [
        ("faithfulness",       faithfulness),
        ("answer_relevancy",   answer_relevancy),
        ("context_precision",  context_precision),
        ("context_recall",     context_recall),
    ]

    logging.info("Running chunked evaluation (5 questions per batch)…")

    from functools import reduce
    all_metric_dfs = []

    for batch_start in range(0, len(dataset), CHUNK_SIZE):
        batch_ds = dataset.select(range(batch_start, min(batch_start + CHUNK_SIZE, len(dataset))))
        batch_tag = f"batch {batch_start // CHUNK_SIZE + 1}"
        logging.info("Processing %s (%d questions)…", batch_tag, len(batch_ds))

        batch_metric_dfs = []
        for m_name, m_obj in metrics_to_evaluate:
            logging.info("  → Metric %s on %s", m_name, batch_tag)
            result = evaluate(
                dataset=batch_ds,
                metrics=[m_obj],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                concurrency=1
            )
            try:
                df_m = result.to_pandas()
            except AttributeError:
                df_m = pd.DataFrame(result.scores)

            metric_cols = [c for c in df_m.columns if c not in ("question", "answer", "contexts", "ground_truth", "ground_truth_context")]
            df_m = df_m[["question"] + metric_cols].rename(columns={metric_cols[0]: m_name})
            batch_metric_dfs.append(df_m)

        # merge metrics for this batch, then append to overall list
        batch_df = reduce(lambda left, right: pd.merge(left, right, on="question"), batch_metric_dfs)
        all_metric_dfs.append(batch_df)

        logging.info("  …%s complete. Cooling down for %d seconds.", batch_tag, COOLDOWN)
        time.sleep(COOLDOWN)

    # --- combine all batches ---
    df = pd.concat(all_metric_dfs, ignore_index=True)

    output_path = "evaluation/evaluation_results.csv"
    df.to_csv(output_path, index=False)
    
    logging.info(f"Successfully saved evaluation results to {output_path}")
    
    summary = df.mean(numeric_only=True)
    print("\n--- Evaluation Summary ---")
    print(summary.to_string())

if __name__ == "__main__":
    main()
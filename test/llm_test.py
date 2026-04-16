import os
import csv

import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
from test_utils import prepare_evaluation_prompt

load_dotenv()

endpoint = "https://endpoint-andrec2419-resource.cognitiveservices.azure.com/"
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = os.getenv("API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

def call_llm_judge(user_prompt, deployment_name):
    """Send a prepared evaluation prompt to Azure OpenAI and return the yes/no judgment."""
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator. Determine whether the provided article is relevant to the topic. "
                    "Focus on the article text and the topic itself. "
                    "Reply ONLY with 'Yes' if they are a good match, or 'No' if they are not."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        max_completion_tokens=5,
        temperature=0.0,
        model=deployment_name
    )

    return response.choices[0].message.content.strip()

def run_evaluation(
    input_file,
    output_file,
    topic,
    k=10,
    driver_uri="bolt://localhost:23034",
    auth=("", ""),
    deployment_name="gpt-4.1-mini"
):
    """Run the judge on the top-k rows from the input CSV and write results incrementally."""

    df = pd.read_csv(input_file)

    # Only evaluate the leading candidates to keep the run bounded and reproducible.
    top_articles = df.head(k).copy()

    with open(output_file, "w", encoding="utf-8", newline="") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(["rank", "article_id", "judge"])

        for rank, (_, row) in enumerate(top_articles.iterrows(), start=1):
            article_id = str(row["node_id"]).strip()

            try:
                # Build the article-specific prompt before sending it to the judge model.
                prompt = prepare_evaluation_prompt(topic, article_id, driver_uri, auth)
                result = call_llm_judge(prompt, deployment_name)
                writer.writerow([rank, article_id, result])
                output_handle.flush()
                print(f"Evaluated rank {rank}: {article_id}, Prompt: {prompt}... -> {result}")
            except ValueError as exc:
                # Preserve per-row failures so one bad prompt does not stop the batch.
                writer.writerow([rank, article_id, f"ERROR: {exc}"])
                output_handle.flush()
                print(f"Error processing rank {rank}, {article_id}: {exc}")
            except Exception as exc:
                # Catch unexpected API or runtime errors and keep processing the remaining rows.
                writer.writerow([rank, article_id, f"ERROR: {exc}"])
                output_handle.flush()
                print(f"Unexpected error on rank {rank}, {article_id}: {exc}")

# def run_evaluation_batch(
#     input_file,
#     output_file,
#     topic,
#     sample_size=10,
#     pool_size=1000,
#     driver_uri="bolt://localhost:23034",
#     auth=("", ""),
#     deployment_name="gpt-4.1-mini"
# ):
#     """Samples article couples, evaluates them via LLM, and logs the results."""
    
#     print(f"Sampling {sample_size} couples from {input_file}...")
#     couples = sample_couples(input_file, sample_size, pool_size)
    
#     # Open the file once before the loop to optimize I/O operations
#     with open(output_file, "a", encoding="utf-8") as f:
#         for article_id, _ in couples:
#             try:
#                 # Build prompt for the LLM
#                 prompt = prepare_evaluation_prompt(topic, article_id, driver_uri, auth)

#                 # Evaluate the prompt with the LLM
#                 result = call_llm_judge(prompt, deployment_name)

#                 # Write the result to file
#                 f.write(f"{article_id}, {result}\n")
                
#                 # Force the write to disk immediately so you can monitor progress
#                 f.flush() 
                
#                 print(f"Evaluated: {article_id} -> {result}")

#             except ValueError as e:
#                 print(f"Error processing {article_id}: {e}")
#             except Exception as e:
#                 # Catch LLM API or network errors so the whole batch doesn't crash
#                 print(f"Unexpected error on {article_id}: {e}")


if __name__ == "__main__":
    run_evaluation(
        input_file="test/test_outputs/combustibili/new_results_combustibili.csv",
        output_file="test/test_outputs/combustibili/llm_judge_output_combustibili.txt",
        topic="Normativa sui combustibili ad uso trazione, uso civile, industriale e marittimo.",
        k=1,
    )

import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from test_utils import prepare_evaluation_prompt, sample_couples

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
    """Takes a pre-formatted prompt, calls Azure OpenAI, and returns the binary judgment."""
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator. Determine if the two provided articles "
                    "share a core topic, regulate similar concepts, or logically enrich each other. "
                    "Pay close attention to their listed Topics and Titles. "
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

def run_evaluation_batch(
    input_file,
    output_file,
    sample_size=10,
    pool_size=1000,
    driver_uri="bolt://localhost:23034",
    auth=("", ""),
    deployment_name="gpt-4.1-mini"
):
    """Samples article couples, evaluates them via LLM, and logs the results."""
    
    print(f"Sampling {sample_size} couples from {input_file}...")
    couples = sample_couples(input_file, sample_size, pool_size)
    
    # Open the file once before the loop to optimize I/O operations
    with open(output_file, "a", encoding="utf-8") as f:
        for article_1_id, article_2_id in couples:
            try:
                # Build prompt for the LLM
                prompt = prepare_evaluation_prompt(article_1_id, article_2_id, driver_uri, auth)

                # Evaluate the prompt with the LLM
                result = call_llm_judge(prompt, deployment_name)

                # Write the result to file
                f.write(f"{article_1_id}, {article_2_id}, {result}\n")
                
                # Force the write to disk immediately so you can monitor progress
                f.flush() 
                
                print(f"Evaluated: {article_1_id} & {article_2_id} -> {result}")

            except ValueError as e:
                print(f"Error processing {article_1_id} and {article_2_id}: {e}")
            except Exception as e:
                # Catch LLM API or network errors so the whole batch doesn't crash
                print(f"Unexpected error on {article_1_id} and {article_2_id}: {e}")


if __name__ == "__main__":
    run_evaluation_batch(
        input_file="output/pairs_combustibili_ranked_batch.csv",
        output_file="test/test_outputs/combustibili/batch_llm_judge_output_combustibili.txt",
        sample_size=10,
        pool_size=1000,
    )

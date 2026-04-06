from neo4j import GraphDatabase
import pandas as pd

# --- DATABASE & PROMPT PREPARATION LOGIC ---

def fetch_article_data(tx, article_id):
    """Executes the Memgraph query to fetch the article text for a single id."""
    query = """
        MATCH (a:LawUnit)<-[r1]-(l:Law)
        WHERE a.id = $article_id
        AND (type(r1) = "HAS_ATTACHMENT" OR type(r1) = "HAS_ARTICLE")
        RETURN a.id AS id, a.text AS text
        """
    result = tx.run(query, article_id=article_id)
    record = result.single()
    return record.data() if record else None

def prepare_evaluation_prompt(topic, article_id, driver_uri, auth):
    """Fetches article text from Memgraph and formats the user prompt string."""
    driver = GraphDatabase.driver(driver_uri, auth=auth)

    with driver.session() as session:
        article_data = session.execute_read(fetch_article_data, article_id)

    if not article_data:
        raise ValueError(f"Missing data in Memgraph for node: {article_id}")

    article_text = str(article_data.get('text', ''))

    prompt = f"""
        Topic: {topic}

        Article Text:
        {article_text}

        Determine whether the article is relevant to the topic.
        """
    return prompt.strip()

# def sample_couples(input_file, num_couples, threshold):
#     """Reads pairs from input CSV, samples a subset of couples from the highest ranked."""
#     df = pd.read_csv(input_file)

#     # Take top 'threshold' rows
#     filtered_df = df.head(threshold).copy() 

#     # skip EU Laws (only national laws)
#     filtered_df = filtered_df[~filtered_df['node_1'].astype(str).str.startswith('3') & ~filtered_df['node_2'].astype(str).str.startswith('3')]
    
#     sampled_df = filtered_df.sample(n=num_couples, random_state=42)
#     return sampled_df[['node_1', 'node_2']].values.tolist()


if __name__ == "__main__":
    df = pd.read_csv("test/test_outputs/combustibili/new_results_combustibili.csv")
    df = df.head(10)
    topic = "Normativa sui combustibili ad uso trazione, uso civile, industriale e marittimo."
    for article_1_id in df['node_id']:
        prompt = prepare_evaluation_prompt(topic, article_1_id, "bolt://localhost:23034", ("", ""))
        print(f"Prompt for {article_1_id}:\n{prompt}")
        print("-" * 80)
        

from neo4j import GraphDatabase
import pandas as pd

# --- DATABASE & PROMPT PREPARATION LOGIC ---

def fetch_article_data(tx, nodes_id):
    """Executes the Memgraph query to fetch article metadata and text."""
    query = """
        MATCH (a:LawUnit)<-[r1]-(l:Law)
        WHERE a.id IN $nodes_id
        AND (type(r1) = "HAS_ATTACHMENT" OR type(r1) = "HAS_ARTICLE")
        RETURN a.id AS id, a.title AS title, a.allTopics AS topics, l.title AS lawTitle, a.text AS text
        """
    result = tx.run(query, nodes_id=nodes_id)
    return {record["id"]: record.data() for record in result}

def prepare_evaluation_prompt(article_1_id, article_2_id, driver_uri, auth, snippet_length=200):
    """Fetches data from Memgraph and formats the user prompt string."""
    driver = GraphDatabase.driver(driver_uri, auth=auth)

    with driver.session() as session:
        nodes_data = session.execute_read(fetch_article_data, [article_1_id, article_2_id])
    
    if article_1_id not in nodes_data or article_2_id not in nodes_data:
        raise ValueError(f"Missing data in Memgraph for nodes: {article_1_id}, {article_2_id}")

    art_1 = nodes_data[article_1_id]
    art_2 = nodes_data[article_2_id]

    # Handle formatting safely for missing fields
    text_1 = str(art_1.get('text', ''))[:snippet_length]
    text_2 = str(art_2.get('text', ''))[:snippet_length]
    topics_1 = ", ".join(art_1.get('topics', [])) if art_1.get('topics') else "None"
    topics_2 = ", ".join(art_2.get('topics', [])) if art_2.get('topics') else "None"

    prompt = f"""
        Please evaluate the correlation between these two articles.

        <Article_1>
        Law Title: {art_1.get('lawTitle', 'Unknown')}
        Article Title: {art_1.get('title', 'Unknown')}
        Topics: {topics_1}
        Text Snippet: {text_1}...
        </Article_1>

        <Article_2>
        Law Title: {art_2.get('lawTitle', 'Unknown')}
        Article Title: {art_2.get('title', 'Unknown')}
        Topics: {topics_2}
        Text Snippet: {text_2}...
        </Article_2>
        """
    return prompt.strip()

def sample_couples(input_file, num_couples, threshold):
    """Reads pairs from input CSV, samples a subset of couples from the highest ranked."""
    df = pd.read_csv(input_file)

    # Take top 'threshold' rows
    filtered_df = df.head(threshold).copy() 

    # skip EU Laws (only national laws)
    filtered_df = filtered_df[~filtered_df['node_1'].astype(str).str.startswith('3') & ~filtered_df['node_2'].astype(str).str.startswith('3')]
    
    sampled_df = filtered_df.sample(n=num_couples, random_state=42)
    return sampled_df[['node_1', 'node_2']].values.tolist()


if __name__ == "__main__":
    couples = sample_couples("output/pairs_combustibili_ranked_full.csv", num_couples=10, threshold=1000)
    for article_1_id, article_2_id in couples:
        print(f"Sampled couple: {article_1_id}, {article_2_id}\n")

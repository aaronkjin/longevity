import os
from dotenv import load_dotenv
import openai
import torch
from transformers import AutoTokenizer, AutoModel
from rdflib import Graph, Literal
from sklearn.metrics.pairwise import cosine_similarity

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# Initialize OpenAI, tokenizer, and model for embeddings
openai.api_key = OPENAI_API_KEY
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


# Get text embeddings
def get_embedding(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings


# Find most similar ontology node + description
def find_most_similar(input_embedding, node_embeddings):
    similarities = cosine_similarity(input_embedding, node_embeddings)
    return similarities.argmax()


# Get node embeddings
def process_user_input(user_input, node_embeddings, node_texts):
    input_embedding = get_embedding(user_input).reshape(1, -1)
    most_similar_node_index = find_most_similar(
        input_embedding, node_embeddings)
    return node_texts[most_similar_node_index]


def main():
    # Load and parse ontology RDF file
    g = Graph()
    g.parse('longevity.rdf', format='application/rdf+xml')

    # Extract texts from RDF graph
    node_texts = [str(o) for s, p, o in g if isinstance(o, Literal)]
    node_embeddings = torch.stack([get_embedding(text) for text in node_texts])

    # Take user input
    user_input = input(
        "Please describe your current lifestyle, including diet, exercise, social activities, etc.: ")

    # Process input to find most similar ontology node + description
    similar_node_text = process_user_input(
        user_input, node_embeddings, node_texts)
    print("Most similar node description:", similar_node_text)

    # Generate prompt with most similar node for GPT
    prompt = f"Based on the ontology node related to: '{similar_node_text}', what are the recommended improvements for the following lifestyle aspects? {user_input}"

    # Generate recommendations
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that hears about users' lifestyles. Based on the given data as well as from your own knowledge, you will recommend to the user some improvements they can have to the lifestyle in terms of living longer."},
            {"role": "user", "content": prompt},
        ]
    )

    # Output recommentation
    recommendation = response['choices'][0]['message']['content']
    print("Recommendation:", recommendation)


if __name__ == "__main__":
    main()

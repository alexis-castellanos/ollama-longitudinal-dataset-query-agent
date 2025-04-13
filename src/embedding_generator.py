# embedding_generator.py

from ollama import Client

def generate_embeddings(text):
    client = Client()
    response = client.embeddings(model="nomic-embed-text", prompt=text)
    return response['embedding']

def prepare_data_for_embeddings(data):
    processed_data = []
    for record in data:
        combined_text = (f"VariableName: {record['variableName']}"
                         f", Description: {record['description']}"
                         f", Question: {record['question']}"
                         f", Response: {record['response']}"
                         )
        embedding = generate_embeddings(combined_text)
        record['embedding'] = embedding
        processed_data.append(record)
    return processed_data


def get_recommendations(user_input, collection):
    user_embedding = generate_embeddings(user_input)
    results = collection.query(
        query_embeddings=[user_embedding],
        n_results=10  # Number of recommendations to return
    )
    print(results)
    return results['documents'], results['ids']
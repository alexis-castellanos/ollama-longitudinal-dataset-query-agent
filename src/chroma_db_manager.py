# chroma_db_manager.py

import chromadb


def initialize_chroma_db():
    """
    Initialize Chroma DB and get or create the 'HRS_data' collection.
    """
    client = chromadb.Client()
    # Use get_or_create_collection to avoid UniqueConstraintError
    collection = client.get_or_create_collection(name="HRS_data")
    return collection

def store_embeddings_in_chroma(data, collection):
    for record in data:
        collection.add(
            embeddings=[record['embedding']],
            documents=[f"VariableName: {record['variableName']}"
                         f", Description: {record['description']}"
                         f", Section: {record['Section']}"
                         f", Level: {record['Level']}"
                         f", Type: {record['Type']}"
                         f", Width: {record['Width']}"
                         f", Decimals: {record['Decimals']}"
                         f", CAI Reference: {record['CAI Reference']}"
                         f", Question: {record['question']}"
                         f", Response: {record['response']}"],
            ids=[record['variableName']]
        )
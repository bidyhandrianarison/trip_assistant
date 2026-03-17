#!/usr/bin/env python
# coding: utf-8
# 1. IMPORTATION DES DONNEES
from dotenv import load_dotenv
from groq import Groq
import os
import json
from qdrant_client import QdrantClient, models


# RECUPERATION DE L'API KEY


load_dotenv()
groq_client = Groq(api_key = os.environ.get('GROQ_API_KEY'))


# CHARGEMENT DES DONNEES:


def load_data(fichier):
    with open(fichier, "rt") as f_in:
        data_raw = json.load(f_in)
    return data_raw

fichier = load_data('travel_data.json')


# CONNEXION A QDRANT ET CREATION DE LA COLLECTION:

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qd_client = QdrantClient(f'http://{qdrant_host}:6333')
model_handle = "jinaai/jina-embeddings-v2-small-en"
EMBEDDING_DIM = 512
collection_name = "travel_assistant"


def prepare_text(entry):
    if entry['type'] == 'destination':
        return f"Destination: {entry['destination']}. Pays: {entry['country']}. Meilleure saison: {entry['best_season']}. Budget par jour: {entry['budget_per_day']}. {entry['description']}"
    else:
        return f"Activité: {entry['activity']} à {entry['destination']}. Catégorie: {entry['category']}. Durée: {entry['duration']}. Coût: {entry['cost']}. {entry['description']}"



# FONCTION DE RECHERCHE ET DE RAG:


def search(query, limit=5):
    results = qd_client.query_points(
        collection_name=collection_name,
        query=models.Document(
            text=query,
            model=model_handle 
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )
    # descriptions = [r.payload['description'] for r in results.points]
    return results


def build_prompt(query, data):
    prompt=f"""
        Tu es un assistant en agence de voyage. A partir des informations suivantes {data},
        Pourrais tu chercher la réponse correspondante à la question de l'utilisateur: {query},
        Si il n'y a pas de réponse correspondante , réponds par : "Je suis désolé, je n'ai pas de réponse à ta question."
        Réponds au voyageur avec un ton calme et généreux.
    """
    return prompt


def call_llm(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def rag(query):
    # 1. Appeler search()
    results = search(query)
    # 2. Construire le prompt
    prompt = build_prompt(query, [r.payload['description'] for r in results.points])
    # 3. Appeler Groq
    answer = call_llm(prompt)
    # 4. Retourner la réponse
    return answer

# FLASK 
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data['question']
    answer = rag(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
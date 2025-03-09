import json
import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from googlesearch import search
import re
import openai
import copy
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING")      
LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT")        
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")        
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT")       
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")           

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

response_schemas = [
    ResponseSchema(
        name="Introduzione", 
        description="Presentazione del contesto del caso, con riferimenti agli aspetti giuridici e fattuali rilevanti. Rimuovere i nomi di persone."
    ),
    ResponseSchema(
        name="I reati contestati", 
        description="Elenco chiaro e dettagliato di tutti i reati contestati, senza includere i codici degli articoli di legge."
    ),
    ResponseSchema(
        name="Motivi del ricorso", 
        description="Descrizione approfondita delle ragioni per cui è stato presentato il ricorso, evidenziando i punti di diritto e di merito sollevati."
    ),
    ResponseSchema(
        name="Decisione della Corte di Cassazione", 
        description="Analisi della decisione della Corte di Cassazione, con spiegazione delle motivazioni adottate e della valutazione dei motivi di ricorso."
    ),
    ResponseSchema(
        name="Conseguenze", 
        description="Elenco dettagliato delle conseguenze derivanti dalla decisione, comprese eventuali sanzioni, spese processuali e altre misure imposte."
    ),
    ResponseSchema(
        name="Sintesi", 
        description="Riassunto chiaro e dettagliato dell'intero processo, escludendo introduzione e conclusione. Evitare di soffermarsi eccessivamente sulla lunghezza e rimuovere i nomi di persone."
    ),
    ResponseSchema(
        name="Conclusione", 
        description="Sintesi finale dell'esito complessivo del procedimento, con un riepilogo delle decisioni e delle loro implicazioni. Rimuovere i nomi di persone."
    )
]


output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

def generate_response(messages):
    response = llm.invoke(messages)
    return response.content.strip()

def process_pdf_agent(pdf_content: str) -> str:
    prompt = f"""
    Analizza il seguente documento PDF e fornisci una sintesi completa e chiara, traducendo il linguaggio giuridico in uno più semplice.
    Utilizza la seguente struttura per la risposta:
    {format_instructions}
    {pdf_content}
    """
    try:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=pdf_content)
        ]
        return generate_response(messages)
    except Exception as e:
        return f"Errore durante l'elaborazione del PDF: {e}"

pdf_folder = "pdf"
results = []

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Elaboro il file: {filename}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        pdf_content = "\n".join([doc.page_content for doc in documents])
        
        response_text = process_pdf_agent(pdf_content)
        
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            json_content = match.group(1)
        else:
            json_content = response_text
        
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError:
            print(f"Errore nel parsing del JSON per il file {filename}")
            continue

        results.append({
            "file": filename,
            "data": data,
            "v1": embeddings_model.embed_documents(data["I reati contestati"])[0],
            "v2": embeddings_model.embed_documents(f'{data["Introduzione"]}\n{data["Sintesi"]}\n{data["Conclusione"]}')[0]
        })

with open("pdfs.json", "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

print("Elaborazione completata. I risultati sono stati salvati in pdfs.json")
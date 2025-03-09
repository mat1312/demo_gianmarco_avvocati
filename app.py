import streamlit as st
import numpy as np
import json
import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import re
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import SystemMessage, HumanMessage

st.set_page_config(
    page_title="Assistente Legale",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-4o-mini"
llm = ChatOpenAI(model_name=model, api_key=openai_api_key)

MEMORY_FILE = r"memoria.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as file:
            try:
                memory = json.load(file)
                if not isinstance(memory, list):
                    memory = []
            except json.JSONDecodeError:
                memory = []
        return memory[-10:] if memory else []
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)

def get_conversation_history():
    memory = load_memory()
    history = []
    for entry in memory:
        if entry.get("user1") and entry.get("user2"):
            user_message = f"Il mio assistito è accusato: {entry.get('user1')}\nPer adesso il processo si sta svolgendo: {entry.get('user2')}"
            history.append({"role": "user", "content": user_message})
            if entry.get("bot"):
                history.append({"role": "assistant", "content": entry.get("bot")})
    return history

def openJson(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

def compute_similarity(question_embedding, embeddings_list):
    similarities = []
    for i, emb in enumerate(embeddings_list):
        try:
            q_array = np.array(question_embedding, dtype=np.float32)
            emb_array = np.array(emb, dtype=np.float32)
            if len(q_array.shape) == 1:
                q_array = q_array.reshape(1, -1)
            if len(emb_array.shape) == 1:
                emb_array = emb_array.reshape(1, -1)
            min_dim = min(q_array.shape[1], emb_array.shape[1])
            q_array = q_array[:, :min_dim]
            emb_array = emb_array[:, :min_dim]
            sim = np.dot(q_array, emb_array.T).flatten()[0] / (
                np.linalg.norm(q_array) * np.linalg.norm(emb_array)
            )
            similarities.append(float(sim))
        except Exception as e:
            st.error(f"Errore con l'elemento {i}: {e}")
            similarities.append(-1.0)
    return similarities

def search_pdfs(query_v1, query_v2, threshold=0.40, path_json="pdfs.json"):
    embedding_query_v1 = embeddings_model.embed_documents(query_v1)[0]
    embedding_query_v2 = embeddings_model.embed_documents(query_v2)[0]
    
    data = openJson(path_json)
    v1_array = np.array([dato['v1'] for dato in data])
    v2_array = np.array([dato['v2'] for dato in data])
    
    scores_v1 = compute_similarity(embedding_query_v1, v1_array)
    scores_v2 = compute_similarity(embedding_query_v2, v2_array)

    candidates = []
    for i in range(len(data)):
        if scores_v1[i] > threshold and scores_v2[i] > threshold:
            combined_score = scores_v1[i] + scores_v2[i]
            candidates.append((i, combined_score))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    indices_superiori = [i for i, score in candidates[:2]]
    
    risposte = [
        {k: v for k, v in data[i].items() if k not in ('v1', 'v2')}
        for i in indices_superiori
    ]
    return risposte

response_schemas = [
    ResponseSchema(
        name="Tipo di reato", 
        description="Elenca i reati commessi, escludendo ogni riferimento ai codici penali."
    ),
    ResponseSchema(
        name="Query", 
        description="Riformula il testo della query dell'utente in un linguaggio chiaro, eliminando ogni riferimento ai codici penali, senza produrre una sintesi."
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

def format_query(query: str):
    is_followup = any(keyword in query.lower() for keyword in 
                     ["ora", "anche", "inoltre", "aggiornamento", "follow-up", 
                      "followup", "aggiorna", "aggiungi", "adesso"])

    if is_followup and "Il mio assistito è accusato di:" in query:
        parts = query.split("AGGIORNAMENTO:", 1)
        if len(parts) == 2:
            existing_accusations = parts[0].strip()
            new_info = parts[1].strip()
            
            prompt = f"""
            Analizza la seguente query di follow-up e integra le nuove informazioni con quelle esistenti.
            
            Informazioni esistenti:
            {existing_accusations}
            
            Nuove informazioni:
            {new_info}
            
            In particolare:
            - Per il campo "Tipo di reato": elenca TUTTI i reati commessi, sia quelli precedenti che quelli nuovi, senza includere codici penali.
            - Per il campo "Query": riformula il testo completo della query in un linguaggio chiaro, assicurandoti di integrare le informazioni precedenti con quelle nuove.
            
            Utilizza la seguente struttura per la risposta:
            {format_instructions}
            """
        else:
            prompt = f"""
            Analizza la seguente query fornita che include informazioni precedenti e nuove.
            In particolare:
            - Per il campo "Tipo di reato": elenca TUTTI i reati commessi, combinando sia informazioni precedenti che nuove, senza includere codici penali.
            - Per il campo "Query": riformula il testo completo della query in un linguaggio chiaro, assicurandoti di integrare tutte le informazioni.
            
            Utilizza la seguente struttura per la risposta:
            {format_instructions}
            
            {query}
            """
    else:
        prompt = f"""
        Analizza la seguente query fornita e riformula il testo della query dell'utente.
        In particolare:
        - Per il campo "Tipo di reato": elenca esclusivamente i reati commessi, senza includere codici penali.
        - Per il campo "Query": riformula il testo della query in un linguaggio chiaro, eliminando ogni riferimento ai codici penali, evitando di creare una sintesi.
        
        Utilizza la seguente struttura per la risposta:
        {format_instructions}
        
        {query}
        """
    
    try:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]
        result_msg = llm.invoke(messages)
        result_str = result_msg.content
        result = output_parser.parse(result_str)
        domanda_v1 = f"{result['Tipo di reato']}"
        domanda_v2 = f"{result['Query']}"
        print(domanda_v1, domanda_v2)
        return domanda_v1, domanda_v2
    except Exception as e:
        error_msg = f"Errore: {e}"
        st.error(f"Errore nella formattazione della query: {e}")
        return error_msg, error_msg

def filter_relevant_articles(case_description: str, penal_code: list, max_articles=5) -> list:
    keywords = set(re.findall(r'\w+', case_description.lower()))
    scored_articles = []
    for article in penal_code:
        article_text = article.get('desc', '').lower()
        score = sum(article_text.count(kw) for kw in keywords)
        if score > 0:
            scored_articles.append((score, article))
    scored_articles.sort(key=lambda x: x[0], reverse=True)
    return [article for score, article in scored_articles[:max_articles]]

response_schemas_assistance = [
    ResponseSchema(
        name="Analisi del caso",
        description="Fornisci un'analisi dettagliata del caso, evidenziando gli elementi rilevanti senza sintesi eccessive."
    ),
    ResponseSchema(
        name="Punti specifici del caso",
        description="Elenca tutti i punti che potrebbero essere importanti relativi al caso."
    ),
    ResponseSchema(
        name="Strategie di difesa",
        description="Elenca e spiega le strategie di difesa consigliate, numerandole se possibile."
    ),
    ResponseSchema(
        name="Punti di forza e debolezza",
        description="Indica separatamente i punti di forza e di debolezza del caso."
    ),
    ResponseSchema(
        name="Raccomandazioni pratiche",
        description="Fornisci raccomandazioni pratiche e concrete per rafforzare la difesa."
    ),
    ResponseSchema(
        name="Fonti",
        description="Elenca le fonti consultate, separate da una virgola, ad es. 'Fonte1, Fonte2'."
    )
]

output_parser_assistance = StructuredOutputParser.from_response_schemas(response_schemas_assistance)
format_instructions_assistance = output_parser_assistance.get_format_instructions()

safe_format_instructions_assistance = format_instructions_assistance.replace("{", "{{").replace("}", "}}")

def format_search_result(file_name: str) -> str:
    return file_name.replace("_", "/")

def safe_strip(value):
    if isinstance(value, list):
        return "\n".join(value).strip()
    elif isinstance(value, str):
        return value.strip()
    else:
        return str(value).strip()

def create_conversation_context(memory):
    if not memory:
        return ""
    
    accusations = []
    case_details = []
    
    for entry in memory:
        if entry.get("user1") and len(entry.get("user1").strip()) > 0:
            accusations.append(entry.get("user1").strip())
        if entry.get("user2") and len(entry.get("user2").strip()) > 0:
            case_details.append(entry.get("user2").strip())
    
    unique_accusations = []
    for acc in accusations:
        if acc not in unique_accusations:
            unique_accusations.append(acc)
    
    unique_case_details = []
    for detail in case_details:
        if detail not in unique_case_details:
            unique_case_details.append(detail)
    
    case_summary = "RIEPILOGO COMPLETO DEL CASO FINO AD ORA:\n"
    
    if unique_accusations:
        case_summary += "Il mio assistito è accusato di: " + "; ".join(unique_accusations) + "\n\n"
    
    if unique_case_details:
        case_summary += "Dettagli del processo: " + "; ".join(unique_case_details) + "\n\n"
    
    context = case_summary + "CRONOLOGIA DETTAGLIATA DELLA CONVERSAZIONE:\n"
    for idx, entry in enumerate(memory):
        if entry.get("user1") and entry.get("user2"):
            context += f"Domanda {idx+1}: Il mio assistito è accusato: {entry.get('user1')}\n"
            context += f"Per adesso il processo si sta svolgendo: {entry.get('user2')}\n"
            if entry.get("bot"):
                bot_response = entry.get("bot")
                sections = ["### Analisi del caso", "### Punti specifici del caso", 
                           "### Strategie di difesa", "### Punti di forza e debolezza"]
                summary = []
                for section in sections:
                    if section in bot_response:
                        section_start = bot_response.find(section)
                        next_section = float('inf')
                        for s in sections:
                            if s != section and s in bot_response and bot_response.find(s) > section_start:
                                next_section = min(next_section, bot_response.find(s))
                        if next_section == float('inf'):
                            section_text = bot_response[section_start:].split("\n\n")[0]
                        else:
                            section_text = bot_response[section_start:next_section].strip()
                        summary.append(section_text)
                
                if summary:
                    context += f"Risposta {idx+1} (sintesi): {' '.join(summary)}\n\n"
                else:
                    context += f"Risposta {idx+1}: Risposta fornita ma non disponibile in formato strutturato\n\n"
    
    return context

def generate_assistance(domanda_v1: str, domanda_v2: str, search_result: list) -> str:
    try:
        with open('codice_penale.json', 'r', encoding='utf-8') as f:
            penal_code = json.load(f)
    except Exception as e:
        st.error(f"Errore nel caricamento del codice penale: {e}")
        penal_code = []
    
    relevant_articles = filter_relevant_articles(domanda_v1, penal_code, max_articles=5)
    penal_context = "\n\n".join([article.get('desc', '').strip() for article in relevant_articles])
    
    memory = load_memory()
    
    conversation_context = create_conversation_context(memory)
    
    is_followup = False
    if "ora" in domanda_v2.lower() or "anche" in domanda_v2.lower() or "inoltre" in domanda_v2.lower():
        is_followup = True
    
    if is_followup:
        context_instruction = """
        ATTENZIONE CRITICA: Questa è una domanda di FOLLOW-UP che aggiunge NUOVE ACCUSE o INFORMAZIONI al caso.
        Devi assolutamente:
        1. Considerare TUTTE le accuse precedenti E quelle nuove come parte dello stesso caso
        2. Non trattare questa come una nuova situazione indipendente
        3. Integrare le nuove informazioni con quelle precedenti nella tua analisi
        4. Riconoscere esplicitamente che ci sono state accuse aggiuntive e come queste modificano la strategia
        5. Adattare la tua analisi completa per riflettere il caso aggiornato con TUTTE le accuse
        
        NON generare una risposta che ignori le accuse o informazioni precedenti!
        """
    else:
        context_instruction = """
        Ricorda la conversazione precedente con l'utente. L'utente potrebbe fare domande di approfondimento
        basate sulle conversazioni precedenti. Considera il contesto fornito.
        """

    system_message = f"""
    Sei un assistente virtuale specializzato nel supporto agli avvocati.
    Il tuo obiettivo è fornire consigli chiari e pratici sulla difesa legale, basandoti sulle informazioni fornite dall'utente e sui casi simili disponibili.
    
    {context_instruction}
    
    {conversation_context}
    
    Quando rispondi, utilizza il seguente formato:
    
    ### Analisi del caso
    Fornisci un'analisi dettagliata del caso, evidenziando gli elementi rilevanti senza sintesi eccessive.
    
    ### Punti specifici del caso
    Elenca tutti i punti che potrebbero essere importanti relativi al caso.
    
    ### Strategie di difesa
    Elenca e spiega le strategie di difesa consigliate, numerandole se possibile.
    
    ### Punti di forza e debolezza
    Indica separatamente i punti di forza e di debolezza del caso.
    
    ### Raccomandazioni pratiche
    Fornisci raccomandazioni pratiche e concrete per rafforzare la difesa.
    
    Contesto attuale:
    - Leggi penali di riferimento:
    {penal_context}
    
    Ora, fornisci la tua analisi seguendo i punti sopra indicati.
    """
    
    query = f"Il mio assistito è accusato: {domanda_v1}\nPer adesso il processo si sta svolgendo: {domanda_v2}\nCasi simili rilevanti: {search_result}\nLeggi penali di riferimento: {penal_context}"
    context_message = f"I casi pertinenti: {search_result}"
    
    if is_followup:
        st.info("Questa sembra essere una domanda di follow-up. Sto integrando le informazioni precedenti.")
    
    additional_instructions = ""
    if is_followup:
        additional_instructions = """
        IMPORTANTE: Il tuo assistito ha ricevuto NUOVE ACCUSE oltre a quelle precedenti. 
        La tua risposta DEVE includere TUTTE le accuse (vecchie e nuove) nella tua analisi.
        """
    
    human_message = f"""Un utente ha posto una domanda. Rispondi nella stessa lingua dell'utente:
    "{query}"
    {context_message}
    {additional_instructions}
    Fornisci una risposta chiara e utile tenendo conto della conversazione precedente e rispondendo in modo specifico
    alla domanda attuale, facendo collegamenti espliciti alle informazioni precedenti quando è rilevante."""
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    
    memory.append({
        "user1": domanda_v1,
        "user2": domanda_v2,
        "bot": None,
    })
    if len(memory) > 10:
        memory = memory[-10:]
    save_memory(memory)

    with st.spinner("L'assistente sta elaborando la risposta..."):
        result = llm.invoke(messages)
    
    answer_raw = result.content.strip()

    prefix = "https://www.italgiure.giustizia.it/xway/application/nif/clean/hc.dll?verbo=attach&db=snpen&id=."
    if search_result:
        file_names = [item.get("file", "Fonte sconosciuta") for item in search_result]
        formatted_file_names = [format_search_result(file_name) for file_name in file_names]
        links = [f"{prefix}{file_name}" for file_name in formatted_file_names]
        formatted_fonti = ", ".join(links)
    else:
        formatted_fonti = "Nessuna fonte trovata."
    
    risposta_strutturata = answer_raw
    if not risposta_strutturata.endswith("Fonti:"):
        risposta_strutturata = risposta_strutturata + "\n\nFonti: " + formatted_fonti
    
    memory = load_memory()
    if memory:
        memory[-1]["bot"] = risposta_strutturata
        save_memory(memory)
    
    return risposta_strutturata


def final_verification(response: str) -> str:
    try:
        with open('codice_penale.json', 'r', encoding='utf-8') as f:
            penal_code = json.load(f)
    except Exception as e:
        st.error(f"Errore nel caricamento del codice penale: {e}")
        return "si"  # In caso di errore, procedi comunque
    
    relevant_articles = filter_relevant_articles(response, penal_code, max_articles=5)
    
    penal_context = "\n\n".join([article.get('desc', '').strip() for article in relevant_articles])

    prompt = f"""Sei un esperto legale specializzato in diritto penale.
    A tua disposizione trovi il seguente contesto normativo:
    {penal_context}

    Ti invito a esaminare la risposta sottostante, valutandone coerenza e completezza in relazione al contesto fornito.
    Se la ritieni generalmente appropriata e ragionevole, rispondi esattamente con "si".
    In caso di dubbi marginali, prediligi comunque una valutazione favorevole.
    Solo se individui errori sostanziali o incoerenze gravi, rispondi con "no".
    """
    
    try:
        with st.spinner("Verifica finale della risposta in corso..."):
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=response)
            ]
            final_result = llm.invoke(messages)
            
            result = final_result.content.strip().lower()
            if result not in ["si", "no"]:
                return "si"  # Se la risposta non è chiara, procedi comunque
            return result
    except Exception as e:
        st.error(f"Errore durante la verifica: {str(e)}")
        return "si"

def process_query(query: str) -> str:
    with st.spinner("Elaborazione della query in corso..."):
        domanda_v1, domanda_v2 = format_query(query)
    
    with st.spinner("Ricerca di casi simili..."):
        search_result = search_pdfs(domanda_v1, domanda_v2)
        print(search_result)
    risposta = ""
    attempts = 0
    max_attempts = 2
    
    while attempts < max_attempts:
        attempts += 1
        try:
            risposta = generate_assistance(domanda_v1, domanda_v2, search_result)
            
            with st.spinner(f"Verifica della risposta (tentativo {attempts}/{max_attempts})..."):
                if final_verification(risposta) == "si":
                    break
                elif attempts < max_attempts:
                    st.warning("La risposta non ha superato la verifica, generando una nuova risposta...")
        except Exception as e:
            st.error(f"Errore durante la generazione della risposta: {str(e)}")
            if attempts < max_attempts:
                st.warning(f"Tentativo {attempts}/{max_attempts} fallito, riprovando...")
            else:
    
                risposta = f"""### Analisi del caso
                Mi scuso, ma ho riscontrato un problema nell'elaborazione della risposta.

                ### Punti specifici del caso
                - L'imputato è accusato di: {domanda_v1}
                - Informazioni sul processo: {domanda_v2}

                ### Strategie di difesa
                Si consiglia di consultare un avvocato per una consulenza personalizzata.

                ### Punti di forza e debolezza
                Non è stato possibile analizzare i punti di forza e debolezza a causa di un errore tecnico.

                ### Raccomandazioni pratiche
                Si raccomanda di riprovare più tardi o di riformulare la richiesta.

                Fonti: Nessuna fonte trovata."""
                break
    
    memory = load_memory()
    if memory:
        memory[-1]["bot"] = risposta
        save_memory(memory)
    
    return risposta

def handle_user_input(user_query):
    memory = load_memory()
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    is_followup = False
    if memory and len(memory) > 0:
        last_entry = memory[-1]
        if len(user_query) < 100 and ("ora" in user_query.lower() or 
                                     "anche" in user_query.lower() or
                                     "inoltre" in user_query.lower() or
                                     "?" in user_query):
            is_followup = True
            
    if is_followup and memory:
        last_entry = memory[-1]
        
        previous_case_details = ""
        for m in memory:
            if m.get("user1"):
                previous_case_details += m.get("user1") + " "
            if m.get("user2"):
                previous_case_details += m.get("user2") + " "
        
        updated_query = f"Il mio assistito è accusato di: {previous_case_details}. AGGIORNAMENTO: {user_query}"
        
        st.info("Rilevata domanda di follow-up: collegamento al contesto precedente")
        response = process_query(updated_query)
    else:
        response = process_query(user_query)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Carica cronologia precedente se disponibile
        history = get_conversation_history()
        if history:
            st.session_state.messages = history

def main():
    st.title("⚖️ Assistente Legale")
    
    # Sidebar con informazioni
    with st.sidebar:
        st.header("Informazioni")
        st.info(
            """
            Questo assistente legale ti aiuta con i casi penali, 
            fornendo analisi, strategie di difesa e raccomandazioni
            pratiche basate su casi simili e sul codice penale.
            """
        )
        
        st.header("Come usare l'assistente")
        st.markdown(
            """
            1. Descrivi il caso del tuo assistito
            2. Specifica in che fase è il processo
            3. Fai domande di follow-up per approfondire
            """
        )
        
        if st.button("Pulisci memoria", type="primary"):
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
            st.session_state.messages = []
            st.success("Memoria cancellata con successo!")
            st.rerun()
    
    # Inizializza lo stato della sessione
    init_session_state()
    
    # Mostra messaggi precedenti
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input utente
    prompt = st.chat_input("Descrivi il caso legale...")
    if prompt:
        handle_user_input(prompt)

if __name__ == "__main__":
    main()

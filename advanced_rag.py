import os
import json
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI Client
llm_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# Initialize Azure Search Client
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

def analyze_query(query: str) -> dict:
    """1. Query Understanding: Analyzes the query to determine type and retrieval strategy."""
    prompt = f"""Analyze the following user query: "{query}"
    Classify it into one of these types: 'simple', 'comparison', 'multi-hop', 'ambiguous', 'greeting'.
    Recommend a search strategy: 'keyword', 'semantic', 'hybrid', or 'none' (if it's a greeting).
    Respond strictly in JSON format: {{"type": "...", "strategy": "..."}}"""
    
    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def rewrite_query(original_query: str, reason: str) -> str:
    """3. Query Rewriting: Improves or simplifies the query for better retrieval."""
    prompt = f"""The original search query "{original_query}" failed because: {reason}. 
    Rewrite this query to be highly effective for a search engine. Remove fluff words.
    Return ONLY the rewritten query string."""
    
    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def retrieve_documents(query: str, strategy: str, top_k: int = 5) -> list:
    """2. Adaptive Retrieval: Fetches documents based on the chosen strategy."""
    print(f"🔍 Executing {strategy.upper()} search for: '{query}'")
    
    try:
        if strategy == "semantic":
            # Semantic Search (Requires semantic configuration in Azure Search)
            results = search_client.search(
                search_text=query, 
                query_type="semantic", 
                semantic_configuration_name="default", 
                top=top_k
            )
        elif strategy == "hybrid":
            # Hybrid (In a real scenario, this involves vector embeddings. 
            # If you don't have embeddings configured, we use semantic as a strong fallback)
            results = search_client.search(
                search_text=query, 
                query_type="semantic",
                top=top_k
            )
        else:
            # Keyword Search (Default)
            results = search_client.search(
                search_text=query, 
                top=top_k
            )
            
        return [{"id": r.get("id", "N/A"), "content": r.get("chunk", "")} for r in results]
        
    except Exception as e:
        print(f"Search Strategy Error: {e}. Falling back to simple keyword search.")
        # Fallback to simple search if semantic fails
        results = search_client.search(search_text=query, top=top_k)
        return [{"id": r.get("id", "N/A"), "content": r.get("chunk", "")} for r in results]

def rerank_results(query: str, docs: list, top_n: int = 3) -> list:
    """4. Re-ranking: Uses LLM to score and select the most relevant chunks."""
    if not docs:
        return []
    
    scored_docs = []
    for doc in docs:
        prompt = f"""Score the relevance of this document to the query on a scale of 1 to 10.
        Query: {query}
        Document: {doc['content']}
        Respond ONLY with a number between 1 and 10."""
        
        try:
            res = llm_client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": prompt}]
            )
            score = int(res.choices[0].message.content.strip())
        except:
            score = 0
            
        doc['score'] = score
        scored_docs.append(doc)
    
    # Sort by score descending and take Top N
    ranked = sorted(scored_docs, key=lambda x: x['score'], reverse=True)[:top_n]
    return [d for d in ranked if d['score'] > 3] # Filter out terrible matches

def check_context_quality(query: str, docs: list) -> bool:
    """5. Retrieval Quality Check: Determines if context is sufficient to answer."""
    if not docs:
        return False
        
    context = "\n".join([d['content'] for d in docs])
    prompt = f"""Query: "{query}"
    Context: {context}
    Is the provided context sufficient to accurately answer the query?
    Respond strictly in JSON format: {{"is_sufficient": true/false}}"""
    
    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("is_sufficient", False)

def generate_grounded_answer(query: str, docs: list) -> str:
    """7. Grounded Answers: Generates final response purely based on context with citations."""
    context = "\n".join([f"[Source ID: {d['id']}] {d['content']}" for d in docs])
    
    prompt = f"""You are a strict, grounded AI assistant. Answer the user's query ONLY using the provided context.
    If the context does not contain the answer, say "I cannot find this information in the knowledge base."
    Always include the [Source ID] in your answer.
    
    Context:
    {context}"""
    
    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

# ==========================================
# MAIN PIPELINE ORCHESTRATOR
# ==========================================
def run_adaptive_rag(user_query: str) -> dict:
    """Orchestrates the entire Advanced RAG process."""
    logs = []
    
    # Step 1: Understand
    analysis = analyze_query(user_query)
    logs.append(f"Query Analysis: {analysis}")

    
    if analysis.get('type') == 'greeting' or 'thank' in user_query.lower() or 'شكرا' in user_query:
        prompt = f"""The user is greeting or saying thank you: '{user_query}'. 
        Respond naturally and politely as an AI assistant in the same language. Keep it brief."""
        
        response = llm_client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "system", "content": prompt}]
        )
        return {
            "final_answer": response.choices[0].message.content,
            "logs": logs + ["Bypassed RAG search for greeting/thanks."],
            "retrieved_docs": []
        }
    
    current_query = user_query
    final_docs = []
    is_sufficient = False
    
    # Adaptive Loop (Max 2 attempts to prevent infinite loops)
    for attempt in range(1, 3):
        logs.append(f"--- Attempt {attempt} ---")
        
        # Step 2: Retrieve
        docs = retrieve_documents(current_query, analysis.get('strategy', 'keyword'), top_k=5)
        
        # Step 4: Re-rank
        ranked_docs = rerank_results(current_query, docs, top_n=3)
        
        # Step 5: Check Quality
        is_sufficient = check_context_quality(user_query, ranked_docs)
        logs.append(f"Context Sufficient? {is_sufficient}")
        
        if is_sufficient:
            final_docs = ranked_docs
            break
        else:
            if attempt == 1:
                # Step 3: Rewrite Query for the next attempt
                current_query = rewrite_query(user_query, "Initial retrieval yielded low-relevance or insufficient context.")
                logs.append(f"Query Rewritten to: '{current_query}'")

    # Step 7: Generate Final Answer
    if not is_sufficient:
        final_answer = "عذراً، بناءً على المستندات المتاحة، لم أتمكن من العثور على معلومات كافية للإجابة على سؤالك." if "Arabic" in user_query else "Based on the available documents, I could not find sufficient information to accurately answer your question."
    else:
        final_answer = generate_grounded_answer(user_query, final_docs)
        
    return {
        "final_answer": final_answer,
        "logs": logs,
        "retrieved_docs": final_docs
    }
# import sys
# import types

# # --- QUICK FIX FOR RAGAS VERTEX AI BUG ---
# dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
# dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
# sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat
# # ----------------------------------------

# import os
# from dotenv import load_dotenv
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# # 1. Load environment variables
# load_dotenv()

# # 2. Setup Azure OpenAI LLM as the evaluator judge
# azure_llm = AzureChatOpenAI(
#     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
#     temperature=0
# )

# # 3. Setup Azure OpenAI Embeddings model
# azure_embeddings = AzureOpenAIEmbeddings(
#     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_deployment="text-embedding-3-small"
# )

# # 4. Prepare evaluation dataset sample with 'reference' added
# data_samples = {
#     "question": ["What are the three types of data, and can you give examples of each?"],
#     "contexts": [
#         [
#             "Structured Data: Refers to data organized in a predefined format, such as databases with rows and columns. Examples include customer information in CRM systems or financial records in spreadsheets.",
#             "Unstructured Data: Represents data without a predefined structure or format, making it challenging to analyze using traditional methods. Examples include social media posts, emails, videos, and images.",
#             "Semi-Structured Data: Contains elements of both structured and unstructured data. It has some organizational properties but lacks a rigid structure, such as XML files or JSON data."
#         ]
#     ],
#     "answer": [
#         "There are three types of data: 1) Structured Data, which is organized in rows and columns like CRM systems. 2) Unstructured Data, which has no format like emails and videos. 3) Semi-Structured Data, which is a mix of both like JSON and XML."
#     ],
#     "ground_truths": [
#         [
#             "Data is categorized into three types: Structured Data (organized in predefined formats like rows/columns, e.g., CRM info, spreadsheets), Unstructured Data (no predefined format, making it hard to analyze, e.g., social media, videos, emails), and Semi-Structured Data (has organizational properties but no rigid structure, e.g., XML, JSON)."
#         ]
#     ],
#     "reference": [
#         "Data is categorized into three types: Structured Data, Unstructured Data, and Semi-Structured Data with respective examples."
#     ]
# }

# # Convert dictionary into a Dataset format
# dataset = Dataset.from_dict(data_samples)

# print("⏳ Evaluating the RAG agent... Please wait (this might take 10-20 seconds)...")

# try:
#     # 5. Run the RAGAS evaluation framework
#     result = evaluate(
#         dataset=dataset,
#         metrics=[
#             faithfulness,
#             answer_relevancy,
#             context_precision,
#             context_recall
#         ],
#         llm=azure_llm,
#         embeddings=azure_embeddings,
#         raise_exceptions=False
#     )

#     print("\n✅ Evaluation Completed!")
#     print("========================")
#     print(result)

# except Exception as e:
#     print("\n❌ Error during evaluation! Check the details below:")
#     print(e)

import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure clients (Azure OpenAI & Search)
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Test dataset containing sample queries and expected ground truths
test_dataset = [
    {
        "query": "What are the three types of data, and can you give examples of each?",
        "ground_truth": "Data is categorized into three types: Structured Data, Unstructured Data, and Semi-Structured Data with respective examples."
    },
    
]

def evaluate_rag_pipeline():
    print("Starting RAG Evaluation Process (Custom LLM-as-a-Judge)...\n")
    results = []

    for item in test_dataset:
        query = item["query"]
        expected_answer = item["ground_truth"]

        # 1. Retrieve context from Azure AI Search
        search_results = search_client.search(search_text=query, top=3)
        retrieved_context = "\n".join([doc.get("content", "") for doc in search_results])

        # 2. Generate answer using the LLM based on retrieved context
        messages = [
            {"role": "system", "content": "You are an AI assistant. Answer strictly based on the provided context."},
            {"role": "user", "content": f"Context:\n{retrieved_context}\n\nQuestion: {query}"}
        ]
        
        response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-mini"),
            messages=messages,
            max_completion_tokens=300
        )
        generated_answer = response.choices[0].message.content

        # 3. Evaluate Faithfulness (LLM-as-a-judge for hallucination check)
        judge_prompt = f"""
        Given the context: '{retrieved_context}'
        And the generated answer: '{generated_answer}'
        Is the generated answer fully derived and supported by the context? Answer only with 'YES' or 'NO'.
        """
        judge_response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-mini"),
            messages=[{"role": "user", "content": judge_prompt}],
            max_completion_tokens=10
        )
        is_faithful = judge_response.choices[0].message.content.strip()

        # 4. Evaluate Answer Relevancy / Ground Truth Match
        relevancy_prompt = f"""
        Given the expected answer: '{expected_answer}'
        And the generated answer: '{generated_answer}'
        Does the generated answer convey the correct core meaning compared to the expected answer? Answer only with 'YES' or 'NO'.
        """
        relevancy_response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-mini"),
            messages=[{"role": "user", "content": relevancy_prompt}],
            max_completion_tokens=10
        )
        is_relevant = relevancy_response.choices[0].message.content.strip()

        results.append({
            "query": query,
            "retrieved_context_found": bool(retrieved_context),
            "generated_answer": generated_answer,
            "is_faithful": is_faithful,
            "is_relevant": is_relevant
        })

    # Print final evaluation report
    for res in results:
        print(f"Query: {res['query']}")
        print(f"Context Retrieved: {res['retrieved_context_found']}")
        print(f"Generated Answer: {res['generated_answer']}")
        print(f"Faithful (No Hallucination): {res['is_faithful']}")
        print(f"Matches Ground Truth: {res['is_relevant']}")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_rag_pipeline()
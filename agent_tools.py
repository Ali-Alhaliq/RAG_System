import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

def rag_search(query: str) -> str:
    """Searches the internal knowledge base documents for relevant information based on the query."""
    try:
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
        search_results = search_client.search(search_text=query, top=3)
        
        context = ""
        for result in search_results:
            context += f"- {result.get('chunk', '')}\n"
            
        return context if context else "No relevant documents found."
    except Exception as e:
        return f"Error executing search: {str(e)}"

def calculate(expression: str) -> str:
    """Performs mathematical calculations. Pass a valid math expression as a string (e.g., '25 - 8')."""
    try:
        allowed_chars = "0123456789+-*/(). "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return str(result)
        else:
            return "Invalid characters in mathematical expression."
    except Exception as e:
        return f"Calculation error: {str(e)}"

def create_support_ticket(issue_description: str, user_email: str = "user@company.com") -> str:
    """Creates a mock support ticket when human assistance is required."""
    ticket_id = "TICKET-98421"
    return f"Success! Support ticket {ticket_id} has been created for issue: '{issue_description}' assigned to {user_email}."

def get_current_datetime() -> str:
    """Returns the current date and time. Useful when the user asks for today's date or current time."""
    now = datetime.now()
    return f"The current date and time is: {now.strftime('%Y-%m-%d %H:%M:%S')}"
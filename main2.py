# main2.py

from langchain_core.messages import HumanMessage, AIMessage
from rag_utils import build_rag_chain

# Initialise ton RAG à l'avance
rag_chain, retriever = build_rag_chain()
chat_history = []

def answer_question(question: str) -> str:
    global chat_history

    result = rag_chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    response_text = result.content if hasattr(result, "content") else str(result)
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response_text))

    return response_text

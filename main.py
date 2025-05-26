#main.py

############################ Importing Libraries ###########################

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings


############################## Loading PDF File ###########################

print("\nStart Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
folder_path = './Test Input/'
filename = 'recette-cookies.pdf'
loader = PyPDFLoader(folder_path + filename, extract_images=True)
docs = loader.load()
print("End Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print("Number of pages in PDF =", len(docs))

############################## Text Splitter #############################

print("\nStart Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)

print("Number of splits =", len(splits))
print("first split =", splits[0].page_content.split()[:10])
print("first split length =", len(splits[0].page_content.split()))
print("first split metadata =", splits[0].metadata)

############################## Embedding #############################
embedding = OllamaEmbeddings(
    model="llama3.1"
)

############################## Storing The Text #############################
print("Start Storing =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
print("Number of vectors in vectorstore =", len(vectorstore))

############################## Retrieiver ###################################

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
print("End Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

############################## LLM Model ###################################

print('Importing LLM...')
llm = ChatOllama(
    model = "llama3",
    temperature = 0.8)

print('Importing LLM done.')

############################### Prompt Template #############################

condense_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

condense_chain = condense_prompt | llm | StrOutputParser()

'''print("\nStart Condense Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
message = condense_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model in machine learning world"),
        ],
        "question": "What does LLM mean?",
    }
)
print("\nEnd Condense Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))'''

################################ QA Prompt Template ###########################

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_chain
    else:
        return input["question"]
    

################################ Formatting docs ####################################

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

################################ RAG Chain ####################################


rag_chain = (
        RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
        | qa_prompt
        | llm
)
chat_history = []


################################# Chat Loop ####################################

print("Bienvenue dans votre assistant RAG. Posez vos questions ! (Tapez 'exit' pour quitter)")

while True:
    question = input("You: ")

    if question.lower() in {"exit", "quit"}:
        print("Fin de la session.")
        break

    try:
        result = rag_chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })

        # ✅ Gérer le cas AIMessage ou str directement
        response_text = result.content if hasattr(result, "content") else str(result)

        # Mettre à jour l'historique de la conversation
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response_text))

        print("Bot:", response_text)

    except Exception as e:
        print("❌ Erreur pendant le traitement :", e)
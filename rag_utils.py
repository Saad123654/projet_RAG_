# rag_utils.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from datetime import datetime


def build_rag_chain():
    print("############################## Loading PDF File ###########################")
    print("\nStart Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    folder_path = './Test Input/'
    filename = 'Best-Chocolate-Chip-Cookie-Recipe-Ever.pdf'
    loader = PyPDFLoader(folder_path + filename, extract_images=True)
    docs = loader.load()
    print("End Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("Number of pages in PDF =", len(docs))

    print("\n############################## Text Splitter #############################")
    print("Start Splitting =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    print("Number of splits =", len(splits))
    print("First split =", splits[0].page_content.split()[:10])
    print("First split length =", len(splits[0].page_content.split()))
    print("First split metadata =", splits[0].metadata)

    print("\n############################## Embedding #############################")
    embedding = OllamaEmbeddings(model="llama3.1")

    print("\n############################## Storing the text #############################")
    print("Start Storing =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    print("Number of vectors in vectorstore =", len(vectorstore._collection.get()['ids']))

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    llm = ChatOllama(model="llama3.1", temperature=0.8)

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

    

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_chain.invoke(input)
        else:
            return input["question"]

    def format_docs(docs):
        print("\n############################## Context Used #############################")
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:300], "...")  # Affiche les 300 premiers caractères
        return "\n\n".join(doc.page_content for doc in docs)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. Use the following context to answer: \n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
        | qa_prompt
        | llm
    )

    return rag_chain, retriever

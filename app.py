import os
import chainlit as cl
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Constants
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_MODEL = "llama3.2"


# Define the RAGApplication class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


@cl.on_message
async def main(message):
    try:
        query = message.content.strip()

        # Check if the vectorstore exists
        if not os.path.exists(CHROMA_PERSIST_DIR):
            await cl.Message(content="Vector database not found. Please create it first.", author="system").send()
            return

        # Load the ChromaDB vectorstore
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

        # Create the retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Define the prompt template for the LLM
        prompt = PromptTemplate(
            template="""You are a friendly assistant for question-answering tasks and talking. if user talks, have conversation with him. it is not necessary that it has to only questions.
            Use the following documents to answer the question. Don't answer if document is irrelevant to the query.
            If you don't know the answer, just say that you don't know. Do not provide hint that you are having stored document, only use it only for your reference.
            Use three sentences maximum and keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:
            """,
            input_variables=["question", "documents"],
        )

        # Initialize the ChatOllama model
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
        )

        # Define the RAG chain
        rag_chain = prompt | llm | StrOutputParser()

        # Initialize the RAG application
        rag_application = RAGApplication(retriever, rag_chain)

        # Run the RAG application
        answer = rag_application.run(query)

        # Send the response to the user
        await cl.Message(content=f"{answer}").send()

    except Exception as e:
        # Handle exceptions gracefully
        await cl.Message(content=f"An error occurred: {str(e)}", author="system").send()

from langchain_openai import ChatOpenAI
import os
import getpass
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

load_dotenv(override=True)

def main():
    MONGODB_CONNECTION_STRING = os.environ.get("ATLAS_CONNECTION_STRING")
    print("MONGODB_CONNECTION_STRING is ",MONGODB_CONNECTION_STRING)
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    else:
        print("OPENAI_API_KEY is already set")

    llm = init_chat_model("gpt-4.1-nano", model_provider="openai")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    client = MongoClient(MONGODB_CONNECTION_STRING)
    db = client["vectordb1"]
    MONGODB_COLLECTION = db["cvdata"]
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "cvdata"

    vector_store = MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )





    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://shilpa.org/",),
        #
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    # Define prompt for question-answering
    # N.B. for non-US LangSmith endpoints, you may need to specify
    # api_url="https://api.smith.langchain.com" in hub.pull.
    prompt = hub.pull("rlm/rag-prompt")


    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str


    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}


    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({"question": "waht is shilpa?","context":[],"answer":""})
    print(response["answer"])

if __name__ == "__main__":
    main()

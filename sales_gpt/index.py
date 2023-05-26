from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def create_vector_db_from_lenses_json(file_path: str) -> Chroma:
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[]',
        text_content=False
    )
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    vector_db = Chroma.from_texts(
        texts=[doc.page_content.replace('{', '').replace('}', '') for doc in documents],
        embedding=embeddings,
        persist_directory="chromadb",
        collection_name="products",
    )
    vector_db.persist()
    return vector_db


def load_vector_db():
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory="chromadb", embedding_function=embeddings, collection_name="products")
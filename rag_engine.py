import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# initialize vector database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="document_qa",
    embedding_function=sentence_transformer_ef
)


def process_pdf(pdf_file):
    """takes the text from the pdf, splits it into chunks, stores in ChromaDB"""
    reader = PdfReader(pdf_file)
    documents = []
    metadatas = []
    ids = []

    # process page by page to track citations
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        # start chunking text into overlapping segments
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"page": page_num + 1})
            ids.append(f"p{page_num + 1}_c{i}")

    # save chunks and vectors to database
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(documents)


def query_document(query_text, n_results=3):
    """returns the top matching chunks for a query."""
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results
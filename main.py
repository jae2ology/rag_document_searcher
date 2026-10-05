import streamlit as st
import ollama
from rag_engine import process_pdf, query_document

st.set_page_config(page_title="Document Q&A Portal", layout="wide")
st.title("Intelligent Document Search")

# file upload on the side
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Indexing document..."):
            num_chunks = process_pdf(uploaded_file)
            st.success(f"Indexed {num_chunks} text chunks into ChromaDB!")

# main question input
user_query = st.text_input("Ask a question about your document:")

if user_query:
    with st.spinner("Searching document & generating answer..."):
        # 1. Retrieve top context chunks
        db_results = query_document(user_query, n_results=3)
        retrieved_chunks = db_results["documents"][0]
        metadatas = db_results["metadatas"][0]

        # Combine retrieved chunks into context block
        context_text = "\n\n".join(
            [f"[Page {meta['page']}]: {doc}" for doc, meta in zip(retrieved_chunks, metadatas)]
        )

        # 2. Prompt LLM with grounded context
        prompt = f"""
        Answer the question using ONLY the provided context below. 
        If the answer is not in the context, state "Information not found in document.
        Do NOT hallucinate answers.
        "
    
        Context:
        {context_text}

        Question: {user_query}
        Answer:
        """

        response = ollama.generate(model="llama3.2", prompt=prompt)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Answer")
            st.write(response["response"])

        with col2:
            st.subheader("Retrieved Sources & Citations")
            for doc, meta in zip(retrieved_chunks, metadatas):
                with st.expander(f"Page {meta['page']} Citation"):
                    st.write(doc)
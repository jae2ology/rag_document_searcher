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

    st.header("Ollama Settings")
    selected_model = st.text_input("Ollama Model Name", value="llama3.2")


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
        system_prompt = f"""
        Answer the question using ONLY the provided context below. 
        If the answer is not in the context, state "Information not found in document.
        Do NOT hallucinate answers.
        "
    
        Context:
        {context_text}

        Question: {user_query}
        Answer:
        """

        try:
            response = ollama.generate(
                model=selected_model,
                prompt=system_prompt,
                options={
                    "temperature": 0.2
                }
            )


            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Answer")
                st.write(response["response"])

            with col2:
                st.subheader("Retrieved Citations")
                for doc, meta in zip(retrieved_chunks, metadatas):
                    with st.expander(f"Page {meta['page']} Source"):
                        st.write(doc)

        except Exception as e:
            st.error(f"Failed to connect to Ollama. Make sure Ollama is running locally! Error: {e}")
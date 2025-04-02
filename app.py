import streamlit as st
import openai
import faiss
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv

load_dotenv()
import os
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# Initialize FAISS Index
def create_faiss_index(docs):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(docs, embeddings)
    return vector_store

# Extract text from PDFs
def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_gpt_response(query, vector_store):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Use the following context to answer the query:
    {context}
    \nQuery: {query}\nAnswer:"""

    # ✅ Use the updated OpenAI API call format
    client = openai.OpenAI()  # Create a client instance

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content  # Correct response parsing

# Streamlit UI
st.title("📄 Custom GPT Assistant")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    text_data = extract_text_from_pdf(uploaded_files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text_data)
    vector_store = create_faiss_index(chunks)
    st.success("PDFs processed and indexed successfully!")
    
    user_query = st.text_input("Ask a question about the documents:")
    if user_query:
        response = get_gpt_response(user_query, vector_store)
        st.write("### Response:")
        st.write(response)

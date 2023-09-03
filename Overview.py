import os
import time
import tempfile
import chardet
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

# Load environment variables
load_dotenv()
secret_key = os.getenv("API_KEY_CHAT")
openai.api_key = secret_key

# Streamlit UI
st.title("💬 Documentation Upload:")

with st.form("my_form"):
    st.write("URLS:")
    url1 = st.text_input("URL 1:")
    url2 = st.text_input("URL 2:")
    url3 = st.text_input("URL 3:")
    url4 = st.text_input("URL 4:")

    st.write("Files:")
    file1 = st.file_uploader("File 1:")
    file2 = st.file_uploader("File 2:")  

    # Every form must have a submit button
    submitted = st.form_submit_button("Submit")

    # URLs
    urlList = [url for url in [url1, url2, url3, url4] if url]  # Only add non-empty URLs

    if submitted:
        try:
            total_data = []

            # Process uploaded files
            for file in [file1, file2]:  # Loop through both files
                if file:
                    temp_dir = tempfile.TemporaryDirectory()
                    temp_file_path = os.path.join(temp_dir.name, file.name)
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(file.read())

                    loader = UnstructuredFileLoader(temp_file_path)
                    data = loader.load()
                    total_data.extend(data)

            # Process URLs
            if urlList:
                loader2 = UnstructuredURLLoader(urls=urlList)
                data2 = loader2.load()
                total_data.extend(data2)

            # Split text
            text_splitter = CharacterTextSplitter(separator='.\n', 
                                                  chunk_size=100, 
                                                  chunk_overlap=20)
            
            docs = text_splitter.split_documents(total_data)
            

            # Create vector store
            embeddings = OpenAIEmbeddings(openai_api_key=secret_key)
            vectorStore_openAI = FAISS.from_documents(docs, embeddings)
            st.session_state.vectorStore = vectorStore_openAI

            st.write("You can now chat with your assistant.")

        except Exception as e:
            st.write(f"An error occurred: {e}")




# https://www.holymary.es/
# https://www.holymary.es/excelencia-academica/
# https://www.holymary.es/bachillerato-britanico-madrid/
# https://www.holymary.es/valores-catolicos/


# https://www.holymary.es/en/
# https://www.holymary.es/en/academic-excellence/
# https://www.holymary.es/en/vith-form-british-madrid/
# https://www.holymary.es/en/catholic-ethos/
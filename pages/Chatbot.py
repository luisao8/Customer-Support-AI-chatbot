import openai
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import SystemMessage
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.vectorstores import FAISS
import time

load_dotenv()

secret_key = os.getenv("API_KEY_CHAT")


openai.api_key = secret_key

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Type your prompt here"):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        llm=OpenAI(openai_api_key=secret_key,temperature=0)
        if "vectorStore" not in st.session_state or st.session_state.vectorStore is None:
            st.warning("Please upload webpages and transcripts for me to learn first!")
        else:    
            vectorStoreFinal = st.session_state.vectorStore
            # system_message = SystemMessage(content="You are a webpage customer Assistant. You will output very polite and informative answers so that the potential clients are happy,")
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStoreFinal.as_retriever())
            prompt = "You are a webpage customer Assistant for a Catholic School. You will output very polite and informative answers so that the potential clients are happy. Now here is the prompt: " + prompt
            response = chain({"question": prompt}, return_only_outputs=True)
            response = response["answer"]

            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


        # response = response['choices'][0]['message']['content']
        # st.session_state.messages.append(response)
        # st.chat_message("assistant").write(response)
    



from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from file_hash import sha256sum

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
    

def main():
    load_dotenv()
    st.set_page_config(page_title='Ask your PDF')
    st.header("Ask your PDF 📖")

    # upload the file
    pdf = st.file_uploader("Upload your PDF file", type=["pdf"])

    # extract the text
    if pdf is not None:
        file_hash = sha256sum(pdf)
        embeddings = OpenAIEmbeddings()

        # check if we have the index in cache
        # if not, create it
        try:
            with st.spinner("Trying to load index from cache..."):
                index = FAISS.load_local("index_cache/" + file_hash, embeddings)
            st.success("Index loaded from cache.", icon="✅")
            print("Index loaded from cache")
        except:
            with st.spinner("Creating index..."):
                text = extract_text_from_pdf(pdf)
                chunks = split_text_into_chunks(text)
                index = FAISS.from_texts(chunks, embeddings)
            st.success("Index created.", icon="✅")
            print("Index created")
            index.save_local("index_cache/" + file_hash)
            print("Index saved to cache")

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if "messages" not in st.session_state:
            st.session_state.messages = []   

        # show user input
        query = st.text_input(
            "Ask me anything",
            placeholder="Ask me anything from {}".format(pdf.name)
        )

        if query:
            with st.spinner("Thinking...🤔"):
                # load the QA chain
                chain = ConversationalRetrievalChain.from_llm(
                    ChatOpenAI(model_name="gpt-3.5-turbo"), 
                    index.as_retriever(), 
                    memory=st.session_state.memory
                )
                with get_openai_callback() as callback:
                    result = chain({"question": query})
                    print(callback)
                    st.success(result["answer"], icon="🤖")
                st.session_state.messages.append("🐵: {}".format(query))
                st.session_state.messages.append("🤖: {}".format(result["answer"]))
        
        with st.expander("Show chat history"):
            if st.session_state.messages != []:
                for message in reversed(st.session_state.messages):
                    st.write(message)    

if __name__ == '__main__':
    main()
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
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
    st.header("Ask your PDF 💬")

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

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = index.similarity_search(user_question)

            # load the QA chain
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as callback:
                response = chain.run(input_documents=docs, question=user_question)
                print(callback)
                st.write(response)

if __name__ == '__main__':
    main()
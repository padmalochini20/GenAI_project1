import os
import time
import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from secret_key import openai_key


# Load your openai - key
os.environ['OPENAI_API_KEY'] = openai_key

# defining llm
llm = OpenAI(temperature=0.7, max_tokens=700)

# Title of the page
st.title('ARTICLE INSIGHTS')
st.sidebar.title('URLs of Articles')

# Information about the progress - initially kept empty
main_placeholder = st.empty()

# Path of faiss database
file_path = 'faiss_store'

# Getting all the inputs
urls = []
for i in range(2):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

# URL processing button
process_url_clicked = st.sidebar.button('Process URLs')



if process_url_clicked:
    # load data
    doc_loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text('Loading documents from URLs .......')
    data = doc_loader.load()

    # split data
    doc_splitter = RecursiveCharacterTextSplitter(
                        separators=['\n\n','\n','.',','],
                        chunk_size=500,
                        chunk_overlap=70)
    main_placeholder.text('Text Splitter started ........')
    docs = doc_splitter.split_documents(data)

    # Create embedding and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text('Embedding Vector started ........')
    vectorstore_openai.save_local(file_path)
    time.sleep(1)

# Getting the query/question from the user
query = main_placeholder.text_input('Ask Your Question: ')
    
if query:
    if os.path.exists(file_path):
        vectorStore = FAISS.load_local(file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore.as_retriever())
        result = chain({"question" : query}, return_only_outputs=True)

        # result will be in a dictionary format -> 'answer' : [], 'sources' : []
        st.header('Answer:')
        st.write(result['answer'])

        # Display sources if available
        sources = result.get('sources',"")

        if sources:
            st.subheader('Sources')
            sources_list = sources.split('\n')
        
            for source in sources_list:
                st.write(source)
                







# import os
# import streamlit as st
# import pickle
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from secret_key import openai_key


# os.environ['OPENAI_API_KEY'] = openai_key


# # Title
# st.title('Article Insights')

# # Getting article URLs 
# st.sidebar.title('Article URLs')

# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")
# file_path = "faiss_store"

# main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)

# if process_url_clicked:
#     # load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...")
#     data = loader.load()
#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=500
#     )
#     main_placeholder.text("Text Splitter...Started...")
#     docs = text_splitter.split_documents(data)
#     # create embeddings and save it to FAISS index
#     embeddings = OpenAIEmbeddings()
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file

#     vectorstore_openai.save_local(file_path)
#     # with open(file_path, "wb") as f:
#     #     pickle.dump(vectorstore_openai, f)

# query = main_placeholder.text_input("Question: ")

# if query:
#     if os.path.exists(file_path):
        
#         vectorstore = FAISS.load_local(file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#         chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#         result = chain({"question": query}, return_only_outputs=True)
#         # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#         st.header("Answer")
#         st.write(result["answer"])

#         # Display sources, if available
#         sources = result.get("sources", "")
#         if sources:
#             st.subheader("Sources:")
#             sources_list = sources.split("\n")  # Split the sources by newline
#             for source in sources_list:
#                 st.write(source)


import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.response.pprint_utils import pprint_response
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.response.pprint_utils import pprint_response
import streamlit as st


load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# Function to handle file upload and save to local folder
def save_uploaded_files(uploaded_files, target_folder):
    for uploaded_file in uploaded_files:
        file_path = os.path.join(target_folder, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved: {file_path}")

# Streamlit app
def main():
    st.title("PDF File Uploader")

    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        target_folder = "data"
        os.makedirs(target_folder, exist_ok=True)

        save_uploaded_files(uploaded_files, target_folder)
        documents=SimpleDirectoryReader("data").load_data()
        index=VectorStoreIndex.from_documents(documents,show_progress=True)
        # st.write(index)
        retriever=VectorIndexRetriever(index=index,similarity_top_k=4)
        postprocessor=SimilarityPostprocessor(similarity_cutoff=0.80)
        query_engine=RetrieverQueryEngine(retriever=retriever,node_postprocessors=[postprocessor])

        input=st.text_input("Give Text input to get info from uploaded pdf file")
        submit=st.button("Submit")
        if submit:
            response=query_engine.query(input)
            st.subheader('The response is ...')
            # pprint_response(response,show_source=True)
            st.write(str(response))

if __name__ == "__main__":
    main()
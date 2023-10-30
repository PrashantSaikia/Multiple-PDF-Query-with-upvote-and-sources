##########################################################################
# NEED TO RUN ONLY ONCE CREATE THE VECTORSTORE IN WEAVIATE CLOUD (WCS).  #
# NO NEED TO RUN AGAIN UNLESS YOU WANT TO CHANGE THE VECTORSTORE SCHEMA. #
# RESUSE THE VECTORESTORE BY REFERENCING THE NAME OF THE SCHEMA CLASS.   #
##########################################################################

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.weaviate import Weaviate
import weaviate, os
from langchain.embeddings import OpenAIEmbeddings

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = weaviate.Client('http://localhost:80')

# Load the documents
doc_loader = DirectoryLoader(
    r'C:\Users\username\Documents\Docs',
    glob='**/*.pdf',
    show_progress=True
)
docs = doc_loader.load()

# Split the docs into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50
)
splitted_docs_list = splitter.split_documents(docs)

# Create schema
classname = 'LangChain2'
# We need to set index_name and vectorizer for the database, 
# otherwise we will not be able to measure text similarities
# langchain is supposed to set this for you, add this if needed
# You just need to do it the very first time setting the class

if client.schema.exists(classname):
    client.schema.delete_class(classname)

class_obj = {
    "class": classname,
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "vectorizeClassName": True
        }
    }  # Or "text2vec-cohere" or "text2vec-huggingface"
}

try:
  # Add the class to the schema
    client.schema.create_class(class_obj)
except:
  print("Class already exists")

embeddings = OpenAIEmbeddings()
# We use 'classname' for index_name and 'text' for text_key
vectorstore = Weaviate(client, classname, "text", embedding=embeddings)

# add text chunks' embeddings to the Weaviate vector database
texts = [d.page_content for d in splitted_docs_list]
metadatas = [d.metadata for d in splitted_docs_list]
vectorstore.add_texts(texts, metadatas=metadatas, embedding=embeddings)

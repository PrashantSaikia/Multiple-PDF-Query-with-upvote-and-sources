##########################################################################
# NEED TO RUN ONLY ONCE CREATE THE VECTORSTORE (SELF HOSTED K8S CLUSTER).#
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
    r'C:\Users\prasanta.saikia\Documents\Documentation GPT\v1.1\Docs',
    glob='**/*.pdf',
    show_progress=True
)
data = doc_loader.load()

print(f"You have {len(data)} documents.")

# Split the docs into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)
docs = splitter.split_documents(data)

# Create schema
classname = 'Chatbot'
# We need to set index_name and vectorizer for the database, 
# otherwise we will not be able to measure text similarities
# langchain is supposed to set this for you, add this if needed
# You just need to do it the very first time setting the class

if client.schema.exists(classname):
    client.schema.delete_class(classname)

class_obj = {
    "class": classname,
    "description": "Documents for chatbot",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {"model": "ada", "type": "text"},
    },
    "properties": [
        {
            "dataType": ["text"],
            "description": "The content of the paragraph",
            "moduleConfig": {
                "text2vec-openai": {
                    "skip": False,
                    "vectorizePropertyName": False,
                }
            },
            "name": "content",
        },
    ],
}

try:
  # Add the class to the schema
    client.schema.create_class(class_obj)
except:
  print("Class already exists")

embeddings = OpenAIEmbeddings()
# We use 'classname' for index_name and 'content' for text_key
vectorstore = Weaviate(client, classname, "content", embedding=embeddings, attributes=["source"])

# add text chunks' embeddings to the Weaviate vector database
text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]
texts, meta = list(zip(*text_meta_pair))
vectorstore.add_texts(texts, meta)

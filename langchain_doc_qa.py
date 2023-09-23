import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# enable to save to disk and reuse model (for repeated queries on same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

# set up an environment where data is either loaded from an existing vector store
# or created fresh from a text file or directory. Use the data in conjunction with llm
# to build conversational retrieval chain
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore=Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader("data.txt") # use this line if you only need data .txt
    #loader = DirectoryLoader("data/") # user this line if you want to use a whole directory
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

# create conversational retreieval chain using ChatOpenAI and create a retriever to handle searching
# within the vector store. Return the top k results
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
) 


# initialize empty list to keep track of conversation history (pairs of q's and a's)
chat_history = []
# start infinite loop
while True:
    # prompt user to enter query prompt if none provided at runtime
    if not query:
        query = input("Prompt: ")
    # exit conditions
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    # feed user's query and chat history into conversational model
    result = chain({"question": query, "chat_history": chat_history})
    # respond with result answer
    print(result['answer'])
    # update chat history
    chat_history.append((query, result['answer']))
    # reset query
    query = None
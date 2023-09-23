This python code uses langchain, openai and a locally hosted text file (or directory)
to create a large language model that uses ChatGPT in conjunction with local data.


The code takes the locally hosted data and creates a vector embedding and then creates 
a conversational retreieval chain using ChatOpenAI and creates a retriever to handle searching within the vector store. Returning the top k results.

The model can then be queried by the user, and ChatGPT can respond with answers that
blend its existing knowledge and the locally hosted data. 

For example, in my data.txt file, I have an article about the New York Mets/Philadelphia Phillies game
from 9/22/2023. If we ask a question like "Tell me about the New York Mets", we
will get a response that uses information from ChatGPT existing knowledge base,
and the local data we fed our model.

You can fill data.txt with any information you want to blend these capabilities. 

To run the code, run 'python langchain_doc_qa.py'
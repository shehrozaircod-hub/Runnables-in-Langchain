from langchain_classic.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. Load the document
loader = TextLoader("doc.txt")
documents = loader.load()

# 2.  Split the Text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. Convert text into embeddings and store in FAISS
vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

# 4. Retreiver (Fetches relevant documents)
retreiver = vector_store.as_retriever()

# 5. Retreive relevant documents
query = "What are the key takeaways from this document?"
retreived_docs = retreiver._get_relevant_documents(query)

# 6. Combine retreived text into a Single Prompt
retreived_text = "\n".join([doc.page_content for doc in retreived_docs])

# 7. Initialize the LLM
llm = ChatOpenAI(model="gpt-5")

# 8. Pass the retreived documents to LLM
prompt = PromptTemplate(
    template=(
        "You are a helpful assistant. Based on the following extracted text from a document, "
        "provide a concise summary of the key takeaways.\n\n"
        "Question: {query}\n\n"
        "Context:\n{retrieved_text}\n\n"
        "Answer concisely:"
    ),
    input_variables=["query", "retrieved_text"],
)


formatted_prompt = prompt.format(query=query, retrieved_text= retreived_text)

result = llm.invoke(formatted_prompt)
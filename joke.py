from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI LLM
model = ChatOpenAI(model="gpt-5", temperature=0.7)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Tell me a joke about {topic}.\n" \
    "Return only the joke without any additional text.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke in simple terms:\n" \
    "{joke}\n",
    input_variables=["joke"]
)

topic = input("Enter a topic for the joke: ")

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": topic})

print("Joke Explanation:", result)
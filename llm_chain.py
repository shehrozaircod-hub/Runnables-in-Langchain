from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}.\n" \
    "Return only the title without any additional text.",
    input_variables=["topic"]
)

topic = input("Enter a topic for the blog title: ")
formatted_prompt = prompt.format(topic=topic)

# Use the LLM directly (invoke is the modern method)
blog_title = llm.invoke(formatted_prompt)

print("Generated Blog Title:", blog_title)
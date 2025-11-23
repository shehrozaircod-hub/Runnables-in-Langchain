from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

# Define a prompt template
prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}.\n" \
    "Return only the title without any additional text.",
    input_variables=["topic"]
)

# Define the input topic
topic = input("Enter a topic for the blog title: ")

# Format the prompt with the input topic
formatted_prompt = prompt.format(topic=topic)

# Generate the blog title using the LLM
blog_title = llm.invoke(formatted_prompt)

# Print the generated blog title
print("Generated Blog Title:", blog_title)
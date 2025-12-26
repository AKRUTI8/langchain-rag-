import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt1 = PromptTemplate(
    input_variables=["cuisine"],
    template="Suggest a restaurant name for {cuisine} food."
)

prompt2 = PromptTemplate(
    input_variables=["name"],
    template="Suggest some menu items for restaurant {name}."
)

chain1 = prompt1 | llm
chain2 = prompt2 | llm

name_response = chain1.invoke({"cuisine": "Mexican"}).content
menu_response = chain2.invoke({"name": name_response})

print(menu_response.content)

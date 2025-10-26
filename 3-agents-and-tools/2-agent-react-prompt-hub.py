from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()


@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and return the result as a string."""
    try:
        result = eval(expression)  # be cautious with eval in production code
    except Exception as e:
        return f"Error: {e}"
    return str(result)


@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in the mock data."""
    data = {
        "Brazil": "Brasília",
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United States": "Washington, D.C.",
    }
    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return "I don't know the capital of that country."


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [calculator, web_search_mock]

client = Client()
prompt = client.pull_prompt("hwchase17/react")

agent = create_agent(model=llm, tools=tools, system_prompt=prompt.template)

result_malta = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the capital of Malta?"}]},
    debug=False,  # Enable verbose output
)
print(result_malta["messages"][-1].content)

# result_brazil = agent.invoke(
#     {"messages": [{"role": "user", "content": "What is the capital of Brazil?"}]},
#     debug=False,  # Enable verbose output
# )
# print(result_brazil["messages"][-1].content)

# result_calculator = agent.invoke(
#     {"messages": [{"role": "user", "content": "How much is 10 + 10?"}]},
#     debug=False,  # Enable verbose output
# )
# print(result_calculator["messages"][-1].content)

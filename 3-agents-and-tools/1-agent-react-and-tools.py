from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
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


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", disable_streaming=True)
tools = [calculator, web_search_mock]

prompt = """
Answer the following questions as best you can. You have access to the following tools.
Only use the information you get from the tools, even if you know the answer.
If the information is not provided by the tools, say you don't know.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for Observation.
- Never search the internet. Only use the tools provided.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

agent = create_agent(model=llm, tools=tools, system_prompt=prompt)

# result_malta = agent.invoke(
#     {"messages": [{"role": "user", "content": "What is the capital of Malta?"}]},
#     debug=False,  # Enable verbose output
# )
# print(result_malta["messages"][-1].content)

result_brazil = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the capital of Brazil?"}]},
    debug=False,  # Enable verbose output
)
print(result_brazil["messages"][-1].content)

# result_calculator = agent.invoke(
#     {"messages": [{"role": "user", "content": "How much is 10 + 10?"}]},
#     debug=False,  # Enable verbose output
# )
# print(result_calculator["messages"][-1].content)

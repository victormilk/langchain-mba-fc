from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n ```{initial_text}```",
)

template_summary = PromptTemplate(
    input_variables=["text_to_summarize"],
    template="Summarize the following text in 4 words:\n ```{text_to_summarize}```",
)

llm_en = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

translate = template_translate | llm_en | StrOutputParser()
pipeline = (
    {"text_to_summarize": translate} | template_summary | llm_en | StrOutputParser()
)

result = pipeline.invoke(
    {
        "initial_text": "LangChain é um framework para desenvolvimento de aplicações de IA."
    }
)

print(result)

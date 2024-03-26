# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

from dotenv import load_dotenv

load_dotenv()

# OpenAI Chat completion
import os
from openai import AsyncOpenAI
import chainlit as cl
from chainlit.playground.providers import ChatOpenAI

from langchain.agents import create_openai_tools_agent
from langchain.agents import Tool
from langchain.agents.agent import AgentExecutor

from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.prompts import ChatPromptTemplate

from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain.tools.retriever import create_retriever_tool

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
    load_query_constructor_runnable,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_experimental.tools import PythonREPLTool

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from utils import read_from_sqlite, load_sqlite

db_uri = "sqlite:///hr_database.db"
# db_uri_ro = "file:hr_database.db?mode=ro"
data_fp = os.getenv("DATA_FP")  # '/opt/ddlfiles/KDS_DEV/DATA/DS/maven/'
hr_fn = "maven_final_synthetic_data.xlsx"
# print('LOADING SQLITE')
hr_df = read_from_sqlite(db_uri)
# hr_df = load_sqlite(data_fp, hr_fn, db_uri)
python_repl = PythonREPLTool()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
tool_description = "Use this tool to gather numeric data and averages"
agent_db = SQLDatabase.from_uri(db_uri)
sql_toolkit = SQLDatabaseToolkit(
    db=agent_db, llm=ChatOpenAI(temperature=0), tool_description=tool_description
)
sql_context = sql_toolkit.get_context()
sql_tools = sql_toolkit.get_tools()

messages = [
    HumanMessagePromptTemplate.from_template("{input}"),
    # AIMessage(content=SQL_FUNCTIONS_SUFFIX),
    AIMessage(
        """ Use search_survey_response to summarize exit survey data. Use the sql tool for quantitative questions by converting natural language to a sql query."""
    ),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate.from_messages(messages)

if os.path.exists("./chroma_db"):
    print("LOADING CHROMA CACHE")
    vectorstore = Chroma(
        persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()
    )
else:
    print("LOADING CHROMA DOCUMENTS")
    documents = DataFrameLoader(
        hr_df, page_content_column="main_quit_reason_text"
    ).load()
    print("EMBEDDING CHROMA DOCUMENTS")
    vectorstore = Chroma.from_documents(
        documents, OpenAIEmbeddings(), persist_directory="./chroma_db"
    )

metadata_field_info = [
    AttributeInfo(
        name="term_year",
        description="The year the employee quit or was terminated, year from 2019 to 2024. NOTE: Only use the 'eq' operator if a specifc year is mentioned",
        type="integer",
    ),
    AttributeInfo(
        name="term_month",
        description="The month the employee quit or was terminated, month from 1 to 12. NOTE: Only use the 'eq' operator if a specifc month is mentioned",
        type="integer",
    ),
    AttributeInfo(
        name="job_title",
        description=f"The job title of the terminated employee, one of {hr_df['job_title'].unique().tolist()}. NOTE: the query will be case sensitive, so match the options.",
        type="string",
    ),
    AttributeInfo(
        name="business_unit",
        description=f"The department or business unit the terminated employee was located in, one of {hr_df['business_unit'].unique().tolist()}. NOTE: the query will be case sensitive, so match the options.",
        type="string",
    ),
    AttributeInfo(
        name="gender",
        description="The gender of the terminated employee, one of male or female",
        type="string",
    ),
    AttributeInfo(
        name="main_quit_reason_text_sentiment",
        description=f"The sentiment of the main quit reason, one of {hr_df['main_quit_reason_text_sentiment'].unique().tolist()}. NOTE: the query will be case sensitive, so match the options.",
        type="string",
    ),
    AttributeInfo(
        name="nps",
        description="The employee net promoter score, integer from 1 to 10",
        type="integer",
    ),
]
document_content_description = "Reason employee is leaving the company"

llm = ChatOpenAI(model="gpt-4-0613", temperature=0)


query_prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor = query_prompt | llm | output_parser
retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
    enable_limit=True,
    search_kwargs={"k": 50},
    verbose=True,
)

retriever_tool = create_retriever_tool(
    retriever,
    name="survey_search",
    description="Use this tool to summarize semantic information on why employees are quitting the company",
)

openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
openai_agent = create_openai_tools_agent(
    openai_llm, sql_tools + [repl_tool] + [retriever_tool], prompt
)

agent_executor = AgentExecutor(
    agent=openai_agent,
    tools=sql_tools + [repl_tool] + [retriever_tool],
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=10,
)


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 250,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):
    # settings = cl.user_session.get("settings")

    question = message.content

    response = agent_executor.invoke({"input": question})
    response_content = response["output"]

    msg = cl.Message(content=response_content)
    await msg.send()

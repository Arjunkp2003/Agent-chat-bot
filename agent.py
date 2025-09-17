import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool

# ---- Wrappers ----
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

duckduckgo = DuckDuckGoSearchAPIWrapper()
search_tool = Tool(
    name="Search",
    func=duckduckgo.run,
    description="Useful for answering questions about current events or general web search."
)

# ---- Streamlit UI ----
st.title('Langchain - Chat with Search')
st.sidebar.title('Settings')

api_key = st.sidebar.text_input('Enter your API key', type='password')

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': 'Hi, I am a chatbot who can search the web. How can I help you?'}
    ]

# Creates st.session_state.messages (a list of dicts) to store conversation history.

# If not already there, initializes with a greeting.

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Handle new user input
if prompt := st.chat_input(placeholder='What is machine learning?'):

# Adds a chat input box.

# If the user types something, it gets stored in prompt.
# Appends the user’s input to the conversation history.
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name='llama-3.1-8b-instant',
        streaming=True
    )

    tools = [search_tool, arxiv_tool, wiki_tool]

    search_engine = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )
    search_engine = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    handle_parsing_errors=True
)


    search_engine = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    
)

# What a CallbackHandler does

# In LangChain, callbacks are hooks that let you see what’s happening inside the agent/LLM while it’s running.

# Instead of just waiting for the final response, the callback can stream intermediate steps (like tool calls, reasoning traces, partial tokens, etc.).
    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # ✅ Only send latest user input, not the whole history
        response = search_engine.run(prompt, callbacks=[st_cb])

        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)


# So the flow is: initialize → user enters session ID → restore old chat → take new input → run LLM → store + display → persist via session_state.

# LLM role

# The LLM is the large language model (like GPT).

# Its job is to generate answers for the user’s queries.

# It looks at the current user input (and sometimes the chat history if you pass it) and produces a response in natural language.

# Agent role

# The Agent acts as the decision-maker.

# Instead of directly answering, it decides what to do with the user’s query:

# Should it just ask the LLM for a direct answer?

# Or should it call an external tool (like Wikipedia, Arxiv, or a database query) to fetch information, and then let the LLM summarize?

# So in your code flow:

# User asks a question in the chatbot.

# The Agent receives that query.

# The agent may either:

# Directly pass the query to the LLM to generate an answer, OR

# Call an external tool (like WikipediaQueryRun) if the query requires outside info.

# The LLM then uses the result (tool output + chat history + user question) to craft the final answer.


# Opens an assistant chat bubble.

# Creates a callback so you see reasoning + tool calls live.

# Runs the agent on the latest user input only (⚠️ stateless here).

# Appends LLM’s response back into history for display.
# ChatGPT said:

# Right now 👉 your chatbot is stateless:

# You are storing the history in st.session_state.messages (so the UI shows it back).

# But when you call the agent with:

# response = search_engine.run(prompt, callbacks=[st_cb])


# you’re only sending the latest user question (prompt), not the previous messages.
# So the LLM cannot actually “see” earlier conversation turns — it only answers the current input.

# ✅ To make it answer based on previous questions:

# You need to feed the conversation history into the LLM.

# There are two common ways in LangChain:

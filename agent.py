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

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Handle new user input
if prompt := st.chat_input(placeholder='What is machine learning?'):
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
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=3   # 🚀 avoids infinite tool loops
    )

    # Build conversation history as input
    conversation = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n"

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # ✅ Pass conversation history, not just latest input
        response = search_engine.invoke(
            {"input": f"{conversation}\nUser: {prompt}\nAssistant:"},
            callbacks=[st_cb]
        )

        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)

# app.py

import uuid
import streamlit as st

from main import graph
from langchain_core.messages import AIMessage


# Step 1: Configure Streamlit
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖"
)

st.title("🤖 AI Research Assistant")

st.write(
    "Explore research papers, analyze methodologies, "
    "identify research gaps, and generate research ideas."
)


# Step 2: Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Unique conversation ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Step 3: Configure agent memory
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# Step 4: Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Step 5: Chat input
user_input = st.chat_input(
    "Ask me anything about your research..."
)


# Step 6: Process conversation
if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Send latest message to agent
    chat_input = {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    # Get agent response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                final_response = ""

                for state in graph.stream(
                    chat_input,
                    config=config,
                    stream_mode="values"
                ):

                    message = state["messages"][-1]

                    # Display only completed AI responses
                    if isinstance(message, AIMessage):

                        # Ignore intermediate tool calls
                        if message.tool_calls:
                            continue

                        if isinstance(message.content, str):
                            final_response = message.content

                        elif isinstance(message.content, list):
                            final_response = "\n".join(
                                block.get("text", "")
                                for block in message.content
                                if isinstance(block, dict)
                                and block.get("type") == "text"
                            )

                if final_response:

                    st.markdown(final_response)

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_response
                    })

                else:
                    st.warning("No response received.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
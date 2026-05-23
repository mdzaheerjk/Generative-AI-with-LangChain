"""Streamlit app

Run this as follows:
> PYTHONPATH=. streamlit run chapter4/streamlit_app.py
"""
import streamlit as st
from langchain_core.messages import HumanMessage

from chapter4.document_loader import DocumentLoader
from chapter4.rag import graph, config, retriever

# Set page configuration
st.set_page_config(page_title="Corporate Documentation Manager", layout="wide")

# Initialize session state for chat history and file management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    print(f"message: {message}")
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Take all uploaded files
docs = retriever.add_uploaded_docs(st.session_state.uploaded_files)


def process_message(message):
    """Assistant response.

    Note: this ignores the previous messages

    There's some better way to stream this:
    for event in graph.stream(
            {"messages": HumanMessage(message)}, config=config
    ):
        print(event.key())
        if event.key == "doc_finalizer":
            for value in event.values():
                yield value["messages"][-1].content

    """
    response = graph.invoke({"messages": HumanMessage(message)}, config=config)
    return response["messages"][-1].content



# Project description using markdown
st.markdown("""
# 📄 CorpDocs with Citations

CorpDocs is your corporate documentation assistant. This tool generates detailed project documentation,
verifies compliance with corporate standards, and integrates human feedback when necessary. Finally,
it retrieves and attaches source citations to the final document.

**Workflow:**
1. **Generate Documentation:** Create an initial draft.
2. **Compliance Check:** Automatically review for adherence to corporate guidelines.
3. **Human Feedback:** If issues are detected, provide corrective feedback.
4. **Finalize Document:** Produce the revised document.
5. **Add Citations:** Append source citations to the document.

If you like this application, please give us a 5-star review on [Amazon](https://amzn.to/3X1xQbn)!
""")


# Create two columns for chat and file management
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat Interface")

    # React to user input
    if user_message := st.chat_input("Enter your message:"):
        # Display user message in chat message container
        with st.chat_message("User"):
            st.markdown(user_message)
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "User", "content": user_message})
        response = process_message(user_message)
        with st.chat_message("Assistant"):
            st.markdown(response)
        # Add response to chat history
        st.session_state.chat_history.append(
            {"role": "Assistant", "content": response}
        )

with col2:
    st.subheader("Document Management")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=list(DocumentLoader.supported_extensions),
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file)

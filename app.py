from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

template = '''
Answer the following question:

Here is the conversation history: {context}

Question: {question}

Answer:
'''

st.set_page_config(
    page_title='Ollama LLM Chatbot',
    page_icon='🤖',
    layout='wide'
)

st.title('Ollama LLM Chatbot')

st.markdown('''
    Web-based chatbot that uses Ollama for running LLMs locally on your machine.  
    
    **Resources:**            
    * Ollama - *tool that allows you to run open-source large language models (LLMs) locally on your machine: [Github](https://github.com/ollama/ollama)*
    * LangChain - *framework for developing applications powered by LLMs: [Website](https://www.langchain.com/)*
    * Gemma2B - *LLM model which is a part of the Gemma lightweight state-of-the-art open models developed by Google DeepMind: [Kaggle](https://www.kaggle.com/models/google/gemma)*
    * MoonDream2 - *a small vision language model designed to run efficiently on edge devices: [HuggingFace](https://huggingface.co/vikhyatk/moondream2)*
    * Phi3 - *models are the most capable and cost-effective small language models (SLMs) available developed by Microsoft: [HuggingFace](https://huggingface.co/docs/transformers/main/en/model_doc/phi3)*
            
    ---
''') 

local_llm_model = st.selectbox('Choose your local LLM', ['gemma2:2b', 'moondream', 'phi3'], index=0)

st.write('---')

model = OllamaLLM(model=local_llm_model)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

if "messages" not in st.session_state:
    st.session_state.messages = []

assistants_names = {
    "gemma2:2b": "Gemma2B",
    "moondream": "MoonDream",
    "phi3": "Phi3"
}

def main():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"I'm {assistants_names[local_llm_model]}, your local LLM! Type a message to start..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        response = chain.invoke({'context': st.session_state.messages, 'question': prompt})
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
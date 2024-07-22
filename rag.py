import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from sentence_transformers import SentenceTransformer
import pinecone
from langchain.prompts import PromptTemplate
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

# load JSON file
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Initialize Pinecone with the given API key and index name
def initialize_pinecone():
    pc = pinecone.Pinecone(
        api_key=pinecone_api_key
    )
    return pc.Index('capture')

# Initialize the SentenceTransformer model
def initialize_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# Generate an embedding for the given text using the provided model
def generate_embedding(model, text):
    return model.encode(text).tolist()

# Generate an embedding for the given text using the provided model
def generate_embedding(model, text):    
    return model.encode(text).tolist()

# Query Pinecone index with the given query
def query_pinecone(index, query, model, top_k=1, namespace=None):
    query_vector = generate_embedding(model, query)
    response = index.query(vector=query_vector, top_k=top_k, namespace=namespace,include_metadata=True)
    return response['matches']

# Generate a response using OpenAI API based on matches from Pinecone
def generate_response_openai(matches, openai_api_key, query):

    template = "Answer the following question based on the given context:\n\n Question is: {question} \n\n Context is: {context}"

    context = ""
    for match in matches:
        metadata = match['metadata']
        context += f"Answer: {metadata['Answer']}\nSection: {metadata['Section']}\nFormRef: {metadata['FormRef']}\nLocation/Asset: {metadata['Location/Asset']}\nAdditional Information: {metadata['Additional Information']}\n\n"
        # prompt += context

    prompt_template = PromptTemplate(
        input_variables = ["question", "context"],
        template = template
    )    

    user_input = {
        "question": query,
        "context": context
    }    

    final_prompt = prompt_template.format(
        question=user_input["question"], 
        context=user_input["context"]
    )
  
    llm = ChatOpenAI(model='gpt-3.5-turbo', 
                     temperature=0.9,
                     openai_api_key = openai_api_key)

    ai_msg = llm.invoke(final_prompt)

    return ai_msg

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

pinecone_api_key = os.getenv('PINECONE_API_KEY')

st.set_page_config(
    page_title='Connected AI',
    page_icon='😀'
)

st.subheader('Connected AI')

chat = ChatOpenAI(
    model_name='gpt-3.5-turbo', 
    temperature=0.5, 
    openai_api_key=openai_api_key
)

if 'message' not in st.session_state:
    st.session_state.message = []

with st.sidebar:
    # system_message = st.text_input(label='System role')
    user_prompt = st.text_input(label='Ask a question:')
    
    # if system_message and not any(isinstance(x, SystemMessage) for x in st.session_state.message):
    #     st.session_state.message.append(
    #         SystemMessage(content=system_message)
    #     )
    
    # st.session_state.message.append(
    #     SystemMessage(content='You are an assistant.')
    # )
    
    # st.write(st.session_state.message)
    
    if user_prompt:
        st.session_state.message.append(
            HumanMessage(content=user_prompt)
        )
    
        with st.spinner('Working on your request ....'):
            try:                
                print('initializing pinecone ...')
                pinecone = initialize_pinecone()
                print('initializing model ...')
                model = initialize_model()
                print('query pinecone ...')
                matches = query_pinecone(pinecone, user_prompt, model, top_k=3, namespace='')
                print('ask chatgpt ...')
                response = generate_response_openai(matches, openai_api_key, user_prompt)
                st.session_state.message.append(
                    AIMessage(content=response.content)
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
if len(st.session_state.message) >= 1:
    if not isinstance(st.session_state.message[0], SystemMessage):
        st.session_state.message.insert(0, SystemMessage(content='You are a helpful assistant.'))

for index, msg in enumerate(st.session_state.message[1:]):
    if index % 2 == 0:
        message(msg.content, is_user=True, key=f'{index}')
    else:
        message(msg.content, is_user=False, key=f'{index}')

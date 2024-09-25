import streamlit as st
import requests
import os, json
from huggingface_hub import model_info


# App title
st.set_page_config(page_title="🚀💻 FlexLLM Server", layout="wide")

# FastAPI server URL
FASTAPI_URL = "http://localhost:8000/generate/"  # Adjust the port if necessary
FINETUNE_URL = "http://localhost:8000/finetuning"

# Initialize session state variables
if 'added_adapters' not in st.session_state:
    st.session_state.added_adapters = []

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def check_model_availability(model_name):
    try:
        info = model_info(model_name)
        return True
    except Exception:
        return False

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    full_prompt = f"{string_dialogue} {prompt_input} Assistant: "
    
    # Send request to FastAPI server
    response = requests.post(FASTAPI_URL, json={"prompt": full_prompt})
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Sidebar
with st.sidebar:
    st.title('🚀 FlexLLM Server')
    page = st.radio("Choose a page", ["Chat", "Finetune"])
    if page == "Chat":
        st.header('🦙 Llama Chatbot')
        # st.success('Using local FastAPI server', icon='✅')
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        st.subheader('Generation parameters')
        max_length = st.sidebar.slider('Max generation length', min_value=64, max_value=4096, value=2048, step=8)
        # selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'], key='selected_model')
        decoding_method = st.sidebar.selectbox('Decoding method', ['Greedy decoding (default)', 'Sampling'], key='decoding_method')
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01, disabled=decoding_method == 'Greedy decoding (default)')
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, disabled=decoding_method == 'Greedy decoding (default)')
        
        # lora_adapter = st.sidebar.text_input('Lora adapter', placeholder='None')
        st.subheader("LoRA Adapters (optional)")
        # Text input for PEFT model ID
        peft_id = st.text_input("Add a LoRA Adapter", placeholder="Enter the Huggingface PEFT model ID")
        # Button to load the adapter
        if st.button("Load Adapter"):
            if peft_id:
                with st.spinner("Checking PEFT availability..."):
                    is_available = check_model_availability(peft_id)
                if is_available:
                    if peft_id not in st.session_state.added_adapters:
                        st.session_state.added_adapters.append(peft_id)
                        st.success(f"Successfully added PEFT: {peft_id}")
                    else:
                        st.warning(f"PEFT {peft_id} is already in the list.")
                else:
                    st.error(f"PEFT {peft_id} is not available on Hugging Face. Please check the ID and try again.")
            else:
                st.warning("Please enter a PEFT Model ID.")
        # Button to remove all adapters
        if st.button("Remove All Adapters"):
            st.session_state.added_adapters = []
            st.success("All adapters have been removed.")
        # Display the list of added adapters
        st.markdown("**Added Adapters:**")
        if st.session_state.added_adapters:
            for adapter in st.session_state.added_adapters:
                st.write(f"- {adapter}")
        else:
            st.write("No adapters added yet.")
        # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
    elif page == "Finetune":
        st.header("🏋️‍♂️ LoRA Finetuning")
        
        # Hugging Face token input
        hf_token = st.text_input("Enter your Hugging Face token:", type="password")
        
        # Dataset selection
        dataset_option = st.radio("Choose dataset source:", ["Upload JSON", "Hugging Face Dataset"])
        
        if dataset_option == "Upload JSON":
            uploaded_file = st.file_uploader("Upload JSON dataset", type="json")
            if uploaded_file is not None:
                dataset = json.load(uploaded_file)
                st.success("Dataset uploaded successfully!")
        else:
            dataset_name = st.text_input("Enter Hugging Face dataset name:")
        
        # Start finetuning button
        if st.button("Start Finetuning"):
            if not hf_token:
                st.error("Please enter your Hugging Face token.")
            elif dataset_option == "Upload JSON" and uploaded_file is None:
                st.error("Please upload a JSON dataset.")
            elif dataset_option == "Hugging Face Dataset" and not dataset_name:
                st.error("Please enter a Hugging Face dataset name.")
            else:
                # Prepare the request data
                request_data = {
                    "token": hf_token,
                    "dataset_source": dataset_option,
                }
                
                if dataset_option == "Upload JSON":
                    request_data["dataset"] = dataset
                else:
                    request_data["dataset_name"] = dataset_name
                
                # Send finetuning request to FastAPI server
                with st.spinner("Finetuning in progress..."):
                    response = requests.post(FINETUNE_URL, json=request_data)
                
                if response.status_code == 200:
                    st.success("Finetuning completed successfully!")
                else:
                    st.error(f"Finetuning failed. Error: {response.status_code} - {response.text}")

if page == "Chat":
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
elif page == "Finetune":
    st.write("Use the sidebar to configure and start finetuning.")
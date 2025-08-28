"""A simple Streamlit app to generate plot ideas using a fine-tuned GPT2 language model."""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
LATEST_CHECKPOINT = "./output/model/checkpoint-1750"

@st.cache_resource
class AppModel:
    """A simple class to load the model and generate text."""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(LATEST_CHECKPOINT)

    def generate(self, user_prompt: str):
        """Generate a book description based on the user prompt."""
        inputs = self.tokenizer(user_prompt, return_tensors="pt").to("cpu")

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        output_string = self.tokenizer.batch_decode(outputs)

        return output_string


model = AppModel()
st.set_page_config(page_title="Book Plot Generator", page_icon="📔", layout="centered")
st.title("📔 Book Plot Generator")
prompt = st.text_area("Enter the beginning of plot...")
clicked = st.button("✨ Generate!")

if clicked and prompt.strip():
    generated_plot = model.generate(prompt)[0]

    chat_message = st.chat_message("assistant")
    chat_message.markdown(generated_plot)

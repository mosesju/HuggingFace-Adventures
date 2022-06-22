import enum
from lib2to3.pgen2 import token
from numpy import True_
import streamlit as st
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead

model_selected = "gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_selected)

@st.cache
def load_model(model_name):
    model = AutoModelWithLMHead.from_pretrained(model_name)
    return model

model = load_model(model_selected)

def infer(input_ids, temperature, top_p, no_repeat_ngrams):
    
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=40,
        temperature=temperature,
        # top_k=top_k,
        top_p=top_p,
        num_beams=5,
        no_repeat_ngrams=no_repeat_ngrams,
        do_sample=True,
        num_return_sequences=1
    )
    return output_sequences


default_value = "See how a modern neural network auto-completes your text 🤗 "
sent = st.text_area("Text", default_value, height = 275)
# Can I make this more user friendly by making it one dial as opposed to however many?
# max_length = st.sidebar.slider("Max Length", min_value = 10, max_value=30, value=20)
temperature = st.sidebar.slider("Temperature", value = 1.4, min_value = 0.1, max_value=2.0, step=0.05)
# top_k = st.sidebar.slider("Top-k", min_value = 0, max_value=5, value = 0)
top_p = st.sidebar.slider("Top-p", min_value = 0.0, max_value=1.0, step = 0.05, value = 0.9)
# num_return_sequences = st.sidebar.number_input('Number of Return Sequences', min_value=1, max_value=5, value=1, step=1)
# num_beams = st.sidebar.slider("Num Beams", min_value = 1, max_value=10, step = 1, value = 3)
no_repeat_ngrams = st.sidebar.slider("No Repeat Ngrams", min_value = 1, max_value=4, step = 1, value = 4)


encoded_prompt = tokenizer.encode(sent, add_special_tokens=False, return_tensors="pt")
if encoded_prompt.size()[-1]==0:
    input_ids = None
else:
    input_ids = encoded_prompt

output_sequences = infer(input_ids, temperature, top_p, no_repeat_ngrams)

for generated_sequenc_idx, generated_sequence in enumerate(output_sequences):
    print(f"=== GENERATED SEQUENCE {generated_sequenc_idx + 1} ===")
    generated_sequences = generated_sequence.tolist()

    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    total_sequence = (
        sent + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
    )

    generated_sequences.append(total_sequence)
    print(total_sequence)

st.write(generated_sequences[-1])
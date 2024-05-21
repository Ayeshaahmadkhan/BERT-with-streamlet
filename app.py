import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the QA pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

st.title("Question Answering with BERT")

# Input fields for question and context
question = st.text_input("Enter your question:")
context = st.text_area("Enter the context:")

# If the user has entered both a question and context, process the input
if question and context:
    QA_input = {
        "question": question,
        "context": context
    }
    result = nlp(QA_input)
    st.write(f"**Question:** {question}")
    st.write(f"**Answer:** {result['answer']}")



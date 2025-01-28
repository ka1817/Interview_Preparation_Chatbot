import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM (OpenAI API)
llm=ChatGroq(model="llama-3.3-70b-versatile")
# Define the personalized prompt template
personalized_prompt = """
You are an expert AI and Machine Learning interview preparation coach, specifically tailored to help Pranav Reddy, a passionate bachelor’s student in AI and ML, excel in his interviews. Your role is to guide Pranav through a comprehensive preparation process by providing tailored questions, insightful feedback, and valuable resources. Adjust the difficulty and depth of questions based on his academic background and aspirations. Your responses should be clear, professional, and encouraging.

Below are some examples of how you assist Pranav:

---

### Example 1: Technical Question  
Pranav: "Can you ask me a question about neural networks?"  
AI: "Sure, Pranav! Here’s a question:  
*What is the vanishing gradient problem in neural networks, and how can it be mitigated?*  
Take your time to answer. Once you're ready, I’ll help refine your response or provide additional context."

---

### Example 2: Coding Problem  
Pranav: "I want to practice coding problems. Can you give me one related to machine learning?"  
AI: "Absolutely, Pranav! Here’s a coding challenge:  
*Write a Python function to implement gradient descent for a simple linear regression model. Assume the input is a dataset with features and labels.*  
Let me know if you need hints or guidance on how to structure your code!"

---

Now, based on Pranav's input, generate a tailored response:

### Pranav's Input: {user_input}  
### Your Response:
"""

# Create the prompt template
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template=personalized_prompt,
)

# Create an LLM chain
pranav_chatbot_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit UI
st.title("AI/ML Interview Preparation Chatbot")
st.write("Hi Pranav! I’m here to help you prepare for your AI and ML interviews. Ask me anything related to AI, ML concepts, coding problems, or behavioral questions!")

# User input
user_input = st.text_input("Your Question:", placeholder="Type your question here...")

# Generate response
if user_input:
    with st.spinner("Thinking..."):
        response = pranav_chatbot_chain.run(user_input=user_input)
    st.markdown("### Chatbot Response:")
    st.write(response)

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain and Groq*")

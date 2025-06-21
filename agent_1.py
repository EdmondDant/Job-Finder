import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

@st.cache_resource
def load_model_and_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("career_vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    pipe = pipeline("text2text-generation", model="career_model", max_new_tokens=50, temperature=0.0, device=-1)
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = PromptTemplate(
        input_variables=["user_input", "retrieved_examples"],
        template="""
You are a career advisor AI. Your job is to assign the best career sector based on a user's input.

Here are some labeled examples:
{retrieved_examples}

Now classify this:
"{user_input}"

Respond only in this format:
sector: <name of best-fit sector>"""
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    return retriever, chain

retriever, chain = load_model_and_vectorstore()

def make_examples(docs):
    return "\n".join([
        f'\"{doc.page_content[:300]}\" → sector: {doc.metadata["category"]}'
        for doc in docs
    ])

st.title("Career Sector Classifier (Offline Model)")
user_query = st.text_area("Describe your skills or job interests:")

if st.button("Classify") and user_query:
    top_docs = retriever.invoke(user_query)
    examples = make_examples(top_docs)
    response = chain.invoke({"user_input": user_query, "retrieved_examples": examples})
    st.subheader("Suggested Sector:")
    st.markdown(response["text"].strip())



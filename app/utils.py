import os
from pathlib import Path

import datarobot as dr
import dotenv
import pandas as pd
import streamlit as st
from deployment_patch import predict
from langchain import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

dr.Deployment.predict = predict

dotenv.load_dotenv()


def read_topics(file):
    with open(file, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        dict_data = {}

        key = ""
        for line in lines:
            line = line.strip()
            if line.endswith(":"):
                key = line[3:-1].strip()
                dict_data[key] = []
            elif line.startswith("-"):
                dict_data[key].append(line[2:].strip())
    return dict_data


def load_history(customer_id):
    files = Path("data/customer_history/").glob(f"{customer_id}*.txt")
    history = []
    for file in files:
        # extract topic and sub_topic from filename
        topic = file.name.split("_")[1]
        sub_topic = file.name.split("_")[2].split(".")[0]
        with open(file, encoding="utf-8") as f:
            history.append(
                {"conversation": f.read(), "topic": topic, "sub_topic": sub_topic}
            )
    return history


def setup_llm(model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=300):
    """
    Setup the LLM model
    :param model_name: The name of the model to use
    :param temperature: The temperature to use
    :param max_tokens: The maximum number of tokens to generate
    :return: The LLM model
    """
    OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
    OPENAI_ORGANIZATION = os.environ["OPENAI_ORGANIZATION"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]
    OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
    OPENAI_DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]

    llm = AzureChatOpenAI(
        deployment_name=OPENAI_DEPLOYMENT_NAME,
        openai_api_type=OPENAI_API_TYPE,
        openai_api_base=OPENAI_API_BASE,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=OPENAI_API_KEY,
        openai_organization=OPENAI_ORGANIZATION,
        model_name=model_name,
        temperature=temperature,
        verbose=True,
        max_retries=0,
        request_timeout=20,
        max_tokens=max_tokens,
    )
    return llm


def init():
    """
    Initialize the app
    :return: None
    """
    llm = setup_llm()
    long_llm = setup_llm(max_tokens=2000)
    llm_gpt4 = setup_llm(model_name="gpt-4")
    long_llm_gpt4 = setup_llm(model_name="gpt-4", max_tokens=2000)

    if "llm" not in st.session_state:
        st.session_state["llm"] = llm
    if "long_llm" not in st.session_state:
        st.session_state["long_llm"] = long_llm
    if "llm_gpt4" not in st.session_state:
        st.session_state["llm_gpt4"] = llm_gpt4
    if "long_llm_gpt4" not in st.session_state:
        st.session_state["long_llm_gpt4"] = long_llm_gpt4
    topics = read_topics("data/topics.txt")
    credit_topics = read_topics("data/credit_topics.txt")

    if "topics" not in st.session_state:
        st.session_state["topics"] = topics
    if "credit_topics" not in st.session_state:
        st.session_state["credit_topics"] = credit_topics

    lc_data = pd.read_csv("data/10K_Lending_Club_Loans.csv")
    lc_data["desc"] = lc_data["desc"].fillna("")
    lc_data["customer_id"] = lc_data.index + 1000000

    if "lc_data" not in st.session_state:
        st.session_state["lc_data"] = lc_data

    important_features = pd.read_csv("data/lending_club_features.csv")
    if "important_features" not in st.session_state:
        st.session_state["important_features"] = important_features

    example_customers = (
        lc_data[important_features["feature_name"].tolist() + ["customer_id"]]
        .iloc[1:50]
        .to_dict(orient="records")
    )

    if "example_customers" not in st.session_state:
        st.session_state["example_customers"] = example_customers


description_template = PromptTemplate(
    template=(
        "The customer (ID: {customer_id}), with the employment title of {emp_title}, resides in the area with the zip "
        "code {zip_code}. They currently have a loan amounting to ${loan_amnt} with a term of {term}. This loan, "
        "described as '{desc}', carries an interest rate of {int_rate}."
        "The purpose of the loan is for '{purpose}', and it's classified under the grade '{grade}' with a sub-grade of "
        "'{sub_grade}'. The title of this loan is '{title}'.\n\n"
        "This customer has an annual income of ${annual_inc}, and their revolving line utilization rate (the amount of "
        "credit they're using relative to all their available revolving credit or their 'revol_util') stands at "
        "{revol_util}.\n\n"
        "Over the last 6 months, they've had {inq_last_6mths} inquiries on their credit report. It's important to note "
        "that too many hard inquiries might negatively impact a credit score."
    ),
    input_variables=[
        "desc",
        "annual_inc",
        "int_rate",
        "title",
        "term",
        "inq_last_6mths",
        "grade",
        "revol_util",
        "purpose",
        "sub_grade",
        "loan_amnt",
        "emp_title",
        "zip_code",
        "customer_id",
    ],
)

history_template = PromptTemplate(
    template=(
        "Give a a fictional conversation between a bank agent and a customer.\n"
        "The customer's description is given as {customer_desc}\n"
        "They are discussing the topic {topic}."
    ),
    input_variables=["customer_desc", "topic"],
)

feature_change_extraction_prompt_template = PromptTemplate(
    template=(
        "Given are the customer with data {customer_dict} and the conversation history \n"
        "{history}"
        "Please identify any changes in the customer's explaining features."
        "{relevant_features}."
        "Please only respond in json, with the format:"
        '{{"`feature_name`": {{"old_value": "`old_value`", "new_value": "`new_value`"}}}}'
        "If nothing of relevance has changed, please respond with an empty json object."
    ),
    input_variables=["customer_dict", "history", "relevant_features"],
)

topic_detection_prompt_template = PromptTemplate(
    template=(
        "Given are the customer with data {customer_dict} and the conversation history\n\n"
        "{history}\n\n"
        "Please identify the topic of the conversation. The list of all possible topics are given by:\n"
        "{topics}\n"
        "Please only respond in json, with the format:\n"
        '{{"`topic`": "`topic`", "sub_topic": "`sub_topic`"}}\n'
        "If no relevant topic has been detected, please respond with an empty json object."
    ),
    input_variables=["customer_dict", "history", "topics"],
)

chat_template = PromptTemplate(
    template=(
        "You are a Bank agent, talking to a customer. The customer has the following attributes {customer_dict}\n"
        "Never ask to confirm the customer ID - we know that already. Feel free to confirm other attributes, but "
        "call out if there is a mismatch that is unexplained. Always look for information that could be wrong, by "
        "mistake.\n\n"
        "In the past, the customer had the following conversations with the Bank:\n"
        "{history}"
        "Your job is it to help the customer with any questions they might have. You can ask the customer if "
        "they have any questions, or you can ask them if they want to talk about a specific topic.\n\n"
        "Only answer with a complete sentence, only say what the Bank agent has to say - never ask for the "
        "customer ID\n"
        "Please never write `Customer:` - only say what the Bank Agent has to say - then stop\n\n"
        "Here is the conversation so far and the question the customer has asked:\n\n"
        "{chat_history}\n\n"
        "If there is information that is conflicting, please call it out. Look out for any information received in "
        "the conversation so far and check for errors.\n"
        "If the customer presents new information: list the new information first, then answer any queries. Let's "
        "answer step by step\n"
        "Only reply with a single sentence"
    ),
    input_variables=["customer_dict", "chat_history", "history"],
)

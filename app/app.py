import os
import streamlit as st
import pandas as pd
import json
import utils
import datarobot as dr
from PIL import Image
from pathlib import Path
import base64

from dotenv import load_dotenv

load_dotenv()


im = Image.open("datarobot.jpg")

st.set_page_config(
    # layout="wide",
    page_icon=im,
    page_title="Customer Support Assistant App",  # edit this for your usecase
    initial_sidebar_state="auto",
    menu_items={"About": "Customer Support Assist app."},
)


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


st.title("Customer Support Assistant")

header_html = """<style>
    #content {
        position: relative;
    }
    #content img {
        position: absolute;
        top: -50px;
        right: -30px;
    }
</style>""" + "<div id='content'><img src='data:image/png;base64,{}' class='img-fluid' width='50'></div>".format(
    img_to_bytes("./image/Robot-icon-blue-eyes_transparent.png")
)
st.markdown(
    header_html,
    unsafe_allow_html=True,
)

# Create a title for your app
st.header(":blue[Lending Club Customer Support Dashboard]")
# select datarobot deployment dropdown list

client = dr.Client(
    token=os.environ["DATAROBOT_API_TOKEN"], endpoint=os.environ["DATAROBOT_ENDPOINT"]
)
deployment = dr.Deployment.get("6478f959a2205947d6f22602")


def clear_history():
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["feature_changes"] = None
    st.session_state["topic"] = None


if not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.text_input(
        "OpenAI API Key",
        key="openai_api_key",
        on_change=lambda: os.environ.setdefault(
            "OPENAI_API_KEY", st.session_state.openai_api_key
        ),
    )
    st.stop()

utils.init()

# select customer dropdown list
customer = st.sidebar.selectbox(
    "Select a customer",
    st.session_state["example_customers"],
    format_func=lambda customer: customer["customer_id"],
    on_change=clear_history,
)

# get customer history
history = utils.load_history(customer["customer_id"])

# show history and topics on the right side
c1, c2 = st.columns([3, 2])


customer_raw_data = st.session_state["lc_data"]
customer_data = customer_raw_data[
    customer_raw_data["customer_id"] == customer["customer_id"]
]
customer_loan_default_prediction = deployment.predict(customer_data, max_explanations=5)
# st.write(customer_loan_default_prediction)
# st.write(customer_loan_default_prediction.to_dict())

with c1:
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    def get_history(payload, length=5):
        chat_history = ""
        for customer_message, bank_message in list(
            zip(
                payload["inputs"]["past_user_inputs"],
                payload["inputs"]["generated_responses"],
            )
        )[-length:]:
            chat_history += (
                f"Customer: {customer_message} \nBank Agent: {bank_message}\n\n"
            )
        chat_history += f"Customer: {payload['inputs']['text']}"
        return chat_history

    def execute_llm(payload):
        chat_history = get_history(payload, length=100)
        # st.write(chat_history)
        query = utils.chat_template.format(
            customer_dict=customer,
            chat_history=chat_history,
            history=history,
        )

        topic_prompt = utils.topic_detection_prompt_template.format(
            customer_dict=customer,
            history=get_history(payload, length=2),
            topics=st.session_state["topics"],
        )
        detected_topics = st.session_state.llm.predict(topic_prompt)
        try:
            detected_topics = json.loads(detected_topics)
        except:
            pass
        st.session_state["topic"] = detected_topics
        feature_change_prompt = utils.feature_change_extraction_prompt_template.format(
            customer_dict=customer,
            history=get_history(payload, 1),
            relevant_features=st.session_state["important_features"],
        )
        feature_changes = st.session_state["long_llm_gpt4"].predict(
            feature_change_prompt
        )
        try:
            feature_changes = json.loads(feature_changes)
        except:
            feature_changes = None
        st.session_state["feature_changes"] = feature_changes
        # st.write(query)
        response = st.session_state["llm_gpt4"].predict(query)
        # st.write(query)
        return {"generated_text": response}

    def get_text():
        input_text = st.text_input(
            "You: ",
            value="",
            key="input",
        )
        return input_text

    user_input = get_text()

    if user_input:
        output = execute_llm(
            {
                "inputs": {
                    "past_user_inputs": st.session_state.past,
                    "generated_responses": st.session_state.generated,
                    "text": user_input,
                },
                "parameters": {"repetition_penalty": 1.33},
            }
        )

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["generated_text"])

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            with st.chat_message(name="Bank Agent", avatar="🤵"):
                st.write(st.session_state["generated"][i])

            with st.chat_message(name="Customer", avatar="👩‍🦰"):
                st.write(st.session_state["past"][i])
with c2:
    if st.session_state.get("feature_changes", None):
        try:
            for feature in st.session_state["feature_changes"].keys():
                # st.write(st.session_state["feature_changes"][feature]["old_value"])
                # st.write(st.session_state["feature_changes"][feature]["new_value"])
                customer_data_dict = customer_data.to_dict()
                customer_data_dict[feature] = st.session_state["feature_changes"][
                    feature
                ]["new_value"]
            new_record = pd.DataFrame.from_dict(customer_data_dict)
            customer_loan_default_prediction_new = deployment.predict(
                new_record, max_explanations=5
            )
            st.write(
                f"**The prediction for a bad loan has changed from** "
                f"{customer_loan_default_prediction.to_dict(orient='records')[0]['positiveClassPrediction']:.2%} to "
                f"{customer_loan_default_prediction_new.to_dict(orient='records')[0]['positiveClassPrediction']:.2%}"
            )
        except Exception as e:
            st.write(e)
    with st.expander("Customer Data"):
        st.write(customer)
    with st.expander("Customer Loan Default Prediction"):
        st.write(customer_loan_default_prediction.to_dict(orient="records")[0])
    with st.expander("Customer History"):
        if len(history) > 0:
            for conversation in history:
                st.markdown(
                    f"**{conversation['topic']}** - **{conversation['sub_topic']}**"
                )
                # with st.expander("Show Conversation"):
                st.write(conversation["conversation"])
                st.write("---")
        else:
            st.write("#### No History")
    st.write("#### Detected Topic")
    st.write(st.session_state.get("topic", "No topic detected yet"))
    st.write("#### Detected Changed Features")
    st.write(st.session_state.get("feature_changes", "No feature changes detected yet"))

    # st.write(st.session_state["past"])
    # st.write(st.session_state["generated"])
    # st.write(st.session_state.get("topic", None))

    # show only the entries where the values differ between the two dicts:

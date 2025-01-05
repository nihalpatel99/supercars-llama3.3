import streamlit as st 
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from supercars import Supercars
from utils import clean_text
def create_streamlit_app(llm, portfolio, clean_text):
    st.title("Car Customer Email Generator")
    url_input = st.text_input("Enter a car", value="https://en.wikipedia.org/wiki/Porsche_918_Spyder")
    submit_button = st.button("Submit")

    if submit_button:
            try:
                loader = WebBaseLoader([url_input])
                data = clean_text(loader.load().pop().page_content)
                supercars.load_supercars()
                jobs = llm.extract_jobs(data)
                for job in jobs:
                    cars = job.get('engine', [])
                    links = supercars.query_cars(cars)
                    email = llm.write_mail(job, links)
                    st.code(email, language='markdown')
            except Exception as e:
                st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    supercars = Supercars()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, supercars, clean_text)

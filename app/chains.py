import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.5, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
        
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the car page of wikipedia website.
            Your job is to extract the car specifications and return them in JSON format containing the 
            following keys: `model`, `engine`,'top speed' `chassis` and `modes`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):    
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    
    
    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
           """
            ### CAR DESCRIPTION:
            {car_description}
            
            ### INSTRUCTION:
            You are Charles, a business executive sales at Monaco Supercars based in Monaco.
            Your company provides high and ultra rare supercars to VIP customers who you reach out to them via email or phone
            Your job is to write a cold email to the client regarding the cars and their specifications you offer above
            Also add the most relevant ones: {link_list}
            Remeber you are Charlesm business executive sales at Monaco Supercars
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"car_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = self._create_llm()

    def _create_llm(self):
        """
        Initialize the language model with the necessary configurations
        """

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")

        return ChatGroq(
            temperature=0, groq_api_key=api_key, model_name="llama-3.1-70b-versatile"
        )

    def extract_jobs(self,cleaned_text):
        """
        Extract Job postings from the scraped website text and return them as structured json
        """

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
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

    def compose_cold_email(self, job, project_names):
        """Compose a cold email to a client based on the job description and portfolio links."""
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:           
            You are Ghulam Mustafa, a recent Computer Science graduate from Sindh Madressatul Islam University, Karachi. You have gained experience working on diverse projects in Machine Learning and Data Science during your internships at VectraCom and Adamjee Life Assurance, where you implemented NLP models, automated business processes, and applied state-of-the-art ML techniques. You possess strong skills in Python, TensorFlow, PyTorch, and various machine learning frameworks, as well as hands-on experience with end-to-end machine learning projects, including facial recognition, sentiment analysis, and model deployment.

            Your job is to write a cold email to the client regarding the job mentioned above, describing your capability in fulfilling their needs in machine learning, data Science, and AI solutions. Additionally, if the job description does not match your area of expertise, mention that the job differs from your previous work experience in areas like machine learning, data science, or artificial intelligence. Finally, indicate that you are open to relocating outside of Karachi for the right opportunity.
            
            Also add the most relevant projects names from the following to showcase Ghulam Mustafa's portfolio: {proj_names}

            Remember you are Ghulam Mustafa, a fresh Computer Science graduate. Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "proj_names": project_names})
        return res.content

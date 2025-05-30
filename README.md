# Resume Analysis Using LangGraph

What Includes the Project:
    - Extract the work experience and education from the raw resume text.
    - Generate the summary using that extracted information.
    - Extract the insights from that generated summary.
    - Generate Interview Question from that insights.

    
## Clone the Repository
`git clone https://github.com/dhruvilrasadiya/Resume-Analysis.git`

`cd Resume-Analysis`


## Create a virtual environment
`python -m venv .venv`


## Activate virtual environment
`source .venv/bin/activate`      # On macOS/Linux

`.venv\Scripts\activate `        # On Windows


## Install all requirments
`pip install -r requirements.txt`


## Set GROQ API KEY in .env file.
- Create .env file in the directory.
- Set GROQ API KEY in that directory in the following way.
    - GROQ_API_KEY = '-----------'
    - ( get the GROQ_API_KEY from the groqcloud. sign in or register with your email and create a new api key)


## API Access
Run the following command to access the Swagger UI for API.


`uvicorn main:app --reload`


## Resume analysis endpoint
POST /analyze-resume

Description:
    Accepts raw resume text, runs it through the LangGraph workflow, and returns:
        - A generated summary
        - A sample interview question
        - A unique thread_id for checkpointing or resuming
Args:
    request (ResumeRequest): Incoming resume text from the client.
Returns:
    ResumeAnalysisResponse: Includes summary and one interview question.


SAMPLE RESUME TEXT - 
"John Smith is a highly motivated software developer with over five years of experience in full-stack development, cloud infrastructure, and agile collaboration. He is passionate about building scalable applications and improving development processes. Since January 2021, John has been working as a Software Engineer at TechNova Solutions in New York, where he has developed and maintained web applications using React, Node.js, and PostgreSQL. He implemented CI/CD pipelines using Jenkins and GitHub Actions, reducing deployment time by 40%, and led a team of four developers in migrating legacy systems to a microservices architecture. Additionally, he integrated third-party APIs for payment processing and user analytics.Prior to this role, John worked as a Junior Developer at CodeBase Inc. in Jersey City from June 2018 to December 2020. In this position, he assisted in developing internal tools using Python and Flask, participated in code reviews and sprint planning meetings, and contributed to writing unit tests and documentation for RESTful APIs.John holds a Bachelor of Science degree in Computer Science from Rutgers University, which he earned in 2018. During his time at Rutgers, he completed coursework in data structures, algorithms, databases, and software engineering. He was also an active member of the Programming Club and was a finalist at HackRU 2017."


## Question generate endpoint
POST /resume-question

Description:
    Resumes question generation based on an already generated resume summary.
Args:
    req (ResumeCheckpointRequest): Includes `thread_id` and optionally the `resume_summary`.
Returns:
    JSONResponse: Contains generated interview questions or error message.


(TAKE THE SAMPLE INPUT FROM THE OUTPUT OF RESUME ANALYSIS ENDPOINT.)

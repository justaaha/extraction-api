import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from pathlib import Path

load_dotenv()
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    skills: list[str]
    salary_range: str | None = None  # most postings don't list one

class ExtractionError(Exception):
    pass

def clean(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text

def extract(job_text: str) -> JobPosting:
    prompt = f"""Extract structured information from the job posting below.
Return ONLY a raw JSON object, no markdown fences, no extra text, with fields:
title, company, location, skills, salary_range.
skills is a list of short strings (technologies and skills mentioned).
salary_range is a string, or null if not stated.

Job posting:
{job_text}
"""
    last_error = None
    for attempt in range(2):  # first try + one retry
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        try:
            return JobPosting.model_validate_json(clean(response.choices[0].message.content))
        except ValidationError as e:
            last_error = e
    raise ExtractionError(f"Model returned invalid output twice: {last_error}")

if __name__ == "__main__":
    job_text = Path("data/postings/08.txt").read_text(encoding="utf-8")
    try:
        print(extract(job_text).model_dump_json(indent=2))
    except ExtractionError as e:
        print(e)
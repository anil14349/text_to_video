from typing import Dict, Any
from transformers import pipeline
import json

class ScriptGenerator:
    def __init__(self, model_name: str = "gpt2"):
        self.summarizer = pipeline("summarization", model=model_name)
        self.script_template = """
        Hi, I'm {name}, and here's a quick overview of my professional journey.
        
        {education_summary}
        
        {experience_summary}
        
        {skills_summary}
        
        {projects_summary}
        
        Thank you for watching!
        """

    def generate_script(self, resume_data: Dict[str, Any]) -> str:
        """Generate video script from parsed resume data."""
        summaries = {
            "education": self._summarize_section(resume_data.get("education", [])),
            "experience": self._summarize_section(resume_data.get("experience", [])),
            "skills": self._summarize_section(resume_data.get("skills", [])),
            "projects": self._summarize_section(resume_data.get("projects", []))
        }
        
        return self.script_template.format(
            name=resume_data.get("personal_info", {}).get("name", ""),
            education_summary=summaries["education"],
            experience_summary=summaries["experience"],
            skills_summary=summaries["skills"],
            projects_summary=summaries["projects"]
        )

    def _summarize_section(self, section_data: list) -> str:
        """Summarize a section of the resume."""
        if not section_data:
            return ""
            
        text = json.dumps(section_data)
        summary = self.summarizer(text, max_length=50, min_length=10)[0]['summary_text']
        return summary 
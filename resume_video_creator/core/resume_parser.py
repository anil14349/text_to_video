from typing import Dict, Any
import spacy
from pathlib import Path
import docx
import PyPDF2
import re

class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sections = {
            "education": ["education", "academic", "qualification"],
            "experience": ["experience", "employment", "work history"],
            "skills": ["skills", "technical skills", "competencies"],
            "projects": ["projects", "achievements", "accomplishments"]
        }

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse resume file and extract structured information."""
        text = self._read_file(file_path)
        doc = self.nlp(text)
        
        return {
            "personal_info": self._extract_personal_info(doc),
            "education": self._extract_section(doc, "education"),
            "experience": self._extract_section(doc, "experience"),
            "skills": self._extract_section(doc, "skills"),
            "projects": self._extract_section(doc, "projects")
        }

    def _read_file(self, file_path: Path) -> str:
        """Read content from PDF or DOCX file."""
        if file_path.suffix.lower() == '.pdf':
            return self._read_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            return self._read_docx(file_path)
        else:
            raise ValueError("Unsupported file format")

    def _read_pdf(self, file_path: Path) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in reader.pages)

    def _read_docx(self, file_path: Path) -> str:
        doc = docx.Document(file_path)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)

    def _extract_personal_info(self, doc) -> Dict[str, str]:
        # Implementation for extracting name, email, phone, etc.
        return {}

    def _extract_section(self, doc, section_type: str) -> list:
        # Implementation for extracting specific sections
        return [] 
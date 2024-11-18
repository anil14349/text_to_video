import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import re
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_resume_dataset(self, resumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform exploratory data analysis on resume dataset."""
        df = pd.DataFrame(resumes)
        
        analysis = {
            "total_resumes": len(df),
            "skills_distribution": self._analyze_skills(df),
            "education_levels": self._analyze_education(df),
            "experience_distribution": self._analyze_experience(df),
            "common_keywords": self._extract_keywords(df)
        }
        
        return analysis

    def generate_visualizations(self, analysis: Dict[str, Any], output_dir: Path):
        """Generate visualizations from analysis results."""
        # Skills word cloud
        self._create_wordcloud(analysis["common_keywords"], output_dir / "skills_wordcloud.png")
        
        # Experience distribution
        self._plot_experience_distribution(
            analysis["experience_distribution"],
            output_dir / "experience_distribution.png"
        )
        
        # Education levels
        self._plot_education_levels(
            analysis["education_levels"],
            output_dir / "education_levels.png"
        )

    def _analyze_skills(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze distribution of skills across resumes."""
        skills = df["skills"].explode().value_counts().to_dict()
        return skills

    def _analyze_education(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze education levels."""
        education_levels = df["education"].apply(self._extract_education_level)
        return education_levels.value_counts().to_dict()

    def _analyze_experience(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze years of experience distribution."""
        experience_years = df["experience"].apply(self._extract_years)
        return experience_years.value_counts().sort_index().to_dict()

    def _extract_keywords(self, df: pd.DataFrame) -> Dict[str, int]:
        """Extract and count important keywords from resumes."""
        text = " ".join(df["skills"].astype(str))
        doc = self.nlp(text.lower())
        keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
        return pd.Series(keywords).value_counts().to_dict()

    def _create_wordcloud(self, keywords: Dict[str, int], output_path: Path):
        """Generate word cloud visualization."""
        wordcloud = WordCloud(width=800, height=400, background_color='white')
        wordcloud.generate_from_frequencies(keywords)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()

    def _plot_experience_distribution(self, experience_dist: Dict[str, int], output_path: Path):
        """Plot experience distribution."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(experience_dist.keys()), y=list(experience_dist.values()))
        plt.title("Distribution of Years of Experience")
        plt.xlabel("Years of Experience")
        plt.ylabel("Count")
        plt.savefig(output_path)
        plt.close()

    def _extract_education_level(self, education: str) -> str:
        """Extract education level from education text."""
        education = education.lower()
        if "phd" in education or "doctorate" in education:
            return "PhD"
        elif "master" in education:
            return "Master's"
        elif "bachelor" in education:
            return "Bachelor's"
        else:
            return "Other"

    def _extract_years(self, experience: str) -> int:
        """Extract years of experience from experience text."""
        years = re.findall(r'(\d+)\s*years?', experience.lower())
        return int(years[0]) if years else 0 
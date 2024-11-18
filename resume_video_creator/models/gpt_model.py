from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
import pandas as pd

class ResumeDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self) -> int:
        return len(self.encodings.input_ids)

class GPTResumeModel:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fine_tune(self, train_texts: List[str], validation_texts: List[str], output_dir: str):
        """Fine-tune the model on resume data."""
        train_dataset = ResumeDataset(train_texts, self.tokenizer)
        val_dataset = ResumeDataset(validation_texts, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

    def generate_script(self, resume_data: Dict[str, Any], max_length: int = 200) -> str:
        """Generate video script from resume data."""
        prompt = self._create_prompt(resume_data)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _create_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Create a prompt for the GPT model based on resume data."""
        return f"""Create a 1-minute video script highlighting the key aspects of this candidate's profile:

Name: {resume_data.get('name', '')}
Education: {resume_data.get('education', '')}
Experience: {resume_data.get('experience', '')}
Skills: {resume_data.get('skills', '')}
Projects: {resume_data.get('projects', '')}

Generate an engaging script that:
1. Introduces the candidate
2. Highlights their key achievements
3. Emphasizes their unique skills
4. Demonstrates their potential value

Script:""" 
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
from typing import List
import tempfile

from ..core.resume_parser import ResumeParser
from ..core.script_generator import ScriptGenerator
from ..core.tts_generator import TTSGenerator
from ..core.video_generator import VideoGenerator

app = FastAPI()

@app.post("/generate-video/")
async def generate_video(
    resume_file: UploadFile = File(...),
    images: List[UploadFile] = File(...)
):
    """Generate video from resume and images."""
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Save uploaded files
            resume_path = temp_dir / resume_file.filename
            with open(resume_path, "wb") as f:
                f.write(await resume_file.read())

            image_paths = []
            for img in images:
                img_path = temp_dir / img.filename
                with open(img_path, "wb") as f:
                    f.write(await img.read())
                image_paths.append(img_path)

            # Process resume
            parser = ResumeParser()
            resume_data = parser.parse_file(resume_path)

            # Generate script
            script_gen = ScriptGenerator()
            script = script_gen.generate_script(resume_data)

            # Generate audio
            tts_gen = TTSGenerator()
            audio_path = temp_dir / "audio.mp3"
            tts_gen.generate_audio(script, audio_path)

            # Generate video
            video_gen = VideoGenerator()
            output_path = temp_dir / "output.mp4"
            video_path = video_gen.create_video(
                audio_path=audio_path,
                images=image_paths,
                output_path=output_path
            )

            # Return video file
            return {"message": "Video generated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
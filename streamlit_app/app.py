import streamlit as st
import requests
from pathlib import Path
import tempfile

def main():
    st.title("Resume Video Creator")
    st.write("Convert your resume into an engaging video presentation!")

    # File uploads
    resume_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    images = st.file_uploader(
        "Upload images for the video", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if st.button("Generate Video") and resume_file and images:
        with st.spinner("Generating your video..."):
            try:
                # Prepare files for API request
                files = {
                    "resume_file": ("resume.pdf", resume_file.getvalue()),
                }
                
                for i, img in enumerate(images):
                    files[f"images"] = (f"image_{i}.jpg", img.getvalue())

                # Make API request
                response = requests.post(
                    "http://localhost:8000/generate-video/",
                    files=files
                )
                
                if response.status_code == 200:
                    st.success("Video generated successfully!")
                    # Add download button or video player here
                else:
                    st.error("Failed to generate video. Please try again.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
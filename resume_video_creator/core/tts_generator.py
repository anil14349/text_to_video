from typing import Optional
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
from gtts import gTTS

class TTSGenerator:
    def __init__(self, azure_key: Optional[str] = None, azure_region: Optional[str] = None):
        self.azure_key = azure_key
        self.azure_region = azure_region
        self.use_azure = azure_key is not None and azure_region is not None

    def generate_audio(self, text: str, output_path: Path) -> Path:
        """Generate audio file from text using either Azure or gTTS."""
        if self.use_azure:
            return self._generate_azure_audio(text, output_path)
        else:
            return self._generate_gtts_audio(text, output_path)

    def _generate_azure_audio(self, text: str, output_path: Path) -> Path:
        """Generate audio using Azure Text-to-Speech."""
        speech_config = speechsdk.SpeechConfig(
            subscription=self.azure_key, 
            region=self.azure_region
        )
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return output_path
        else:
            raise Exception("Audio generation failed")

    def _generate_gtts_audio(self, text: str, output_path: Path) -> Path:
        """Generate audio using Google Text-to-Speech."""
        tts = gTTS(text=text, lang='en')
        tts.save(str(output_path))
        return output_path 
from typing import List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip, VideoClip
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

class VideoStyleTransfer(nn.Module):
    """Simple style transfer network for video frames."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class VideoGenerator:
    def __init__(
        self,
        output_resolution: Tuple[int, int] = (1920, 1080),
        style_transfer: bool = True
    ):
        self.resolution = output_resolution
        self.fps = 30
        self.transition_duration = 0.5
        self.style_transfer = style_transfer
        if style_transfer:
            self.style_model = VideoStyleTransfer()
            if torch.cuda.is_available():
                self.style_model = self.style_model.cuda()

    def create_video(
        self,
        audio_path: Path,
        images: List[Path],
        output_path: Path,
        background_video: Optional[Path] = None,
        avatar_video: Optional[Path] = None
    ) -> Path:
        """Create video with images, transitions, audio, and optional avatar."""
        # Load audio
        audio = AudioFileClip(str(audio_path))
        duration = audio.duration

        # Process images
        clips = []
        time_per_image = (duration - (len(images) - 1) * self.transition_duration) / len(images)

        for img_path in images:
            # Apply style transfer if enabled
            if self.style_transfer:
                styled_img = self._apply_style_transfer(img_path)
                img_clip = (ImageClip(np.array(styled_img))
                           .set_duration(time_per_image)
                           .resize(self.resolution))
            else:
                img_clip = (ImageClip(str(img_path))
                           .set_duration(time_per_image)
                           .resize(self.resolution))
            clips.append(img_clip)

        # Add transitions
        final_clips = []
        for i, clip in enumerate(clips):
            final_clips.append(clip)
            if i < len(clips) - 1:
                transition = self._create_transition(clips[i], clips[i + 1])
                final_clips.append(transition)

        # Combine clips
        main_video = concatenate_videoclips(final_clips)
        
        # Add avatar if provided
        if avatar_video:
            avatar_clip = (VideoFileClip(str(avatar_video))
                         .resize(width=self.resolution[0] // 4)
                         .set_duration(duration))
            
            # Position avatar in bottom right corner
            avatar_pos = (
                self.resolution[0] - avatar_clip.w - 50,
                self.resolution[1] - avatar_clip.h - 50
            )
            
            main_video = CompositeVideoClip([
                main_video,
                avatar_clip.set_position(avatar_pos)
            ])

        # Add audio
        final_video = main_video.set_audio(audio)

        # Write output file
        final_video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec='libx264',
            audio_codec='aac'
        )

        return output_path

    def _apply_style_transfer(self, image_path: Path) -> Image.Image:
        """Apply style transfer to image."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self._preprocess_image(img)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        with torch.no_grad():
            styled_tensor = self.style_model(img_tensor)
        
        return self._postprocess_image(styled_tensor)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for style transfer."""
        img = np.array(image)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img = img.unsqueeze(0) / 255.0
        return img

    def _postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert processed tensor back to image."""
        img = tensor.cpu().squeeze().numpy()
        img = (img * 255).clip(0, 255).transpose((1, 2, 0)).astype(np.uint8)
        return Image.fromarray(img)

    def _create_transition(self, clip1, clip2, transition_type: str = "fade"):
        """Create transition between two clips."""
        if transition_type == "fade":
            return clip1.crossfadein(self.transition_duration)
        elif transition_type == "slide":
            return self._create_slide_transition(clip1, clip2)
        return clip1

    def _create_slide_transition(self, clip1, clip2):
        """Create slide transition effect."""
        def make_frame(t):
            # Calculate position based on time
            pos = t / self.transition_duration
            frame1 = clip1.get_frame(clip1.duration - t)
            frame2 = clip2.get_frame(t)
            
            # Slide effect
            h, w = frame1.shape[:2]
            offset = int(w * pos)
            
            result = np.zeros_like(frame1)
            result[:, :w-offset] = frame1[:, offset:]
            result[:, w-offset:] = frame2[:, :offset]
            
            return result

        return VideoClip(make_frame, duration=self.transition_duration) 
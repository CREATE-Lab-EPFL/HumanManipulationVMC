"""
Extract audio from a video file recorded with the Canon camera.

Usage:
    python extract_audio.py <video_file>

Output: same filename with .m4a extension (AAC audio, no re-encoding).
"""

import subprocess
import sys
import os

if len(sys.argv) != 2:
    print('Usage: python extract_audio.py <video_file>')
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.isfile(video_path):
    print(f'File not found: {video_path}')
    sys.exit(1)

base, _ = os.path.splitext(video_path)
audio_path = base + '.m4a'

subprocess.run([
    'ffmpeg', '-i', video_path,
    '-vn',           # drop video
    '-acodec', 'copy',  # copy audio stream without re-encoding
    audio_path,
], check=True)

print(f'Audio saved to: {audio_path}')

"""
python script/video_preprocessing.py <video dir>
"""
import os, glob, sys

DIR = sys.argv[1]

videos = glob.glob(f"{DIR}/videos/*.mp4")
videos.sort()

# make the frames dataset
os.system(f"mkdir {DIR}/frames")

ffmpeg_cmd = "ffmpeg -i {video_name} {target}"

for video_path in videos:
  video_name = video_path[video_path.rfind('/'):-4]
  frame_dir = f"{DIR}/frames/{video_name}"
  os.system(f"mkdir {frame_dir}")
  os.system(ffmpeg_cmd.format(
    video_name=video_path,
    target=f"{frame_dir}/%04d.png"))

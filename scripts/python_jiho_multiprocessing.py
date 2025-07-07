import ffmpeg
import os

def split_video(input_path, duration, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    probe = ffmpeg.probe(input_path)
    total_duration = float(probe['format']['duration'])

    segment_paths = []
    i = 0
    while i * duration < total_duration:
        output_path = os.path.join(output_dir, f"segment_{i:03d}.mp4")
        ffmpeg.input(input_path, ss=i*duration, t=duration).output(output_path).run()
        segment_paths.append(output_path)
        i += 1
    return segment_paths

def convert_segments(segments, new_ext=".mkv"):
    converted = []
    for path in segments:
        base, _ = os.path.splitext(path)
        new_path = base + new_ext
        ffmpeg.input(path).output(new_path).run()
        converted.append(new_path)
    return converted

def concatenate_segments(segment_paths, output_path):
    # ffmpeg concat은 텍스트 리스트를 파일로 만들어서 사용해야 함
    with open("file_list.txt", "w") as f:
        for path in segment_paths:
            f.write(f"file '{path}'\n")
    ffmpeg.input("file_list.txt", format='concat', safe=0).output(output_path, c='copy').run()
    os.remove("file_list.txt")

# 예시 사용
input_video = "input.mp4"
output_dir = "segments"
duration = 10  # 10초마다 자르기

segments = split_video(input_video, duration, output_dir)
converted = convert_segments(segments, new_ext=".mkv")
concatenate_segments(converted, "final_output.mkv")
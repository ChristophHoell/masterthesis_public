import os
from argparse import ArgumentParser
from loguru import logger

ffmpeg_location = "/usr/bin/ffmpeg"


def combine_videos(args):
    video_names = os.listdir(args.input_path)
    num_videos = len(video_names)

    videos = []
    row = []
    col = []
    id_vid = 0
    id_row = 0
    for i, v in enumerate(video_names):
        col.append(v)

        if len(col) == args.n_col:
            row.append(col)
            col = []

        if len(row) == args.n_row:
            videos.append(row)
            row = []

    if col != []:
        row.append(col)
    if row != []:
        videos.append(row)
    
    for i, v in enumerate(videos):
        for j, r in enumerate(v):
            for k, c in enumerate(r):
                if k == 0:
                    prev_c = c
                else:
                    out_c = f"tmp_col_{i}_{j}_{k}.mp4"
                    os.system(f"{args.ffmpeg_location} -i {args.input_path}{prev_c} -i {args.input_path}{c} -filter_complex hstack -c:v libx264 -preset slow -crf 5 -c:a aac -movflags +faststart {args.input_path}{out_c} -y -loglevel quiet")
                    if k > 1:
                        os.system(f"rm {args.input_path}{prev_c}")
                        pass
                    prev_c = out_c
            if j == 0:
                prev_r = out_r = out_c
            else:
                out_r = f"tmp_row_{i}_{j}.mp4"
                os.system(f"{args.ffmpeg_location} -i {args.input_path}{prev_r} -i {args.input_path}{out_c} -filter_complex '[1][0]scale2ref=iw:ow/mdar[2nd][ref];[ref][2nd]vstack[vid]' -map [vid] -c:v libx264 -preset slow -crf 5 -c:a aac -movflags +faststart {args.input_path}{out_r} -y -loglevel quiet")
                os.system(f"rm {args.input_path}{prev_r}")
                os.system(f"rm {args.input_path}{out_c}")
                prev_r = out_r

        os.system(f"mv {args.input_path}{out_r} {os.path.join(args.input_path, '..', f'final_{i}.mp4')}")
        os.system(f"mv {args.input_path}{prev_r} {os.path.join(args.input_path, '..', f'final_{i}.mp4')}")

                   





def main():
    parser = ArgumentParser()
    parser.add_argument("--input_path", required = True, type = str, help = "Path to the folder containing the videos to be combined")
    parser.add_argument("--n_row", default = 4, type = int, help = "Max number of videos per row")
    parser.add_argument("--n_col", default = 4, type = int, help = "Max number of videos per column")
    parser.add_argument("--ffmpeg_location", default = "/usr/bin/ffmpeg", type = str, help = "Path to the ffmpeg binary to be used for the combination")
    
    args = parser.parse_args()

    combine_videos(args)

















if __name__ == "__main__":
    main()
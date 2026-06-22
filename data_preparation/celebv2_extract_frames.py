import os
import csv
import random
import cv2
import numpy as np
from tqdm import tqdm

# config...

DATASET_ROOT = '../datasets/video_celeb_v2'
OUTPUT_ROOT = '../datasets/celebv2_frames'

REAL_FOLDER = 'Celeb-real'
FAKE_FOLDER = 'Celeb-synthesis'

NUM_REAL_VIDEOS = 300
FRAMES_PER_VIDEO = 10

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# helpers...

def get_videos(folder):
    
    return sorted([f for f in os.listdir(folder) if f.endswith('.mp4')])


def build_fake_lookup(fake_videos):

    lookup = {}

    for fake in fake_videos:
        parts = fake.replace('.mp4', '').split('_')

        if len(parts) != 3:
            continue

        source_id = parts[0]
        clip_id = parts[2]

        real_name = f'{source_id}_{clip_id}.mp4'

        lookup.setdefault(real_name, [])
        lookup[real_name].append(fake)

    return lookup

# timestamp generation..

def generate_diverse_indices(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if total <= 0:
        return None

    segments = np.linspace(0, total,num_frames + 1, dtype=int)

    indices = []

    for i in range(num_frames):
        start = segments[i]
        end = segments[i + 1]

        if start < end:
            idx = random.randint(start, end - 1)
            indices.append(idx)

    return sorted(indices)

# frame extraction...

def extract_specific_frames(video_path,output_dir, frame_indices, prefix, metadata_rows,label):

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    target_set = set(frame_indices)

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id in target_set:
            local_idx = frame_indices.index(frame_id)
            filename = f'{prefix}_{local_idx:02d}.png'

            save_path = os.path.join(output_dir, filename)

            cv2.imwrite(save_path, frame)

            metadata_rows.append([
                label,
                prefix,
                frame_id,
                filename
            ])

        frame_id += 1

    cap.release()

# main...

def main():

    real_path = os.path.join(DATASET_ROOT, REAL_FOLDER)
    fake_path = os.path.join(DATASET_ROOT, FAKE_FOLDER)

    real_videos = get_videos(real_path)
    fake_videos = get_videos(fake_path)

    print(f'\ntotal real videos: {len(real_videos)}')
    print(f'\ntotal fake videos: {len(fake_videos)}')

    fake_lookup = build_fake_lookup(fake_videos)

    # to keep only real videos that have fake..

    valid_real = [vid for vid in real_videos if vid in fake_lookup]
    print(f'\nvalid real videos: {len(valid_real)}')

    selected_real = random.sample(valid_real, NUM_REAL_VIDEOS)

    metadata_rows = []

    real_count = 0
    fake_count = 0

    # process..

    for real_vid in tqdm(selected_real, desc='Processing Celeb-DF'):

        # real video..

        real_video_path = os.path.join(real_path, real_vid)
        timestamps = generate_diverse_indices(real_video_path, FRAMES_PER_VIDEO)

        if timestamps is None:
            continue

        real_prefix = (
            real_vid
            .replace('.mp4', '')
        )

        extract_specific_frames(
            real_video_path,
            os.path.join(
                OUTPUT_ROOT,
                'real'
            ),
            timestamps,
            prefix=f'real_{real_prefix}',
            metadata_rows=metadata_rows,
            label='real'
        )

        real_count += len(timestamps)

        # fake counterpart..

        fake_vid = sorted(fake_lookup[real_vid])[0]

        fake_video_path = os.path.join(fake_path, fake_vid)

        extract_specific_frames(
            fake_video_path,
            os.path.join(
                OUTPUT_ROOT,
                'fake'
            ),
            timestamps,
            prefix=f"fake_{fake_vid.replace('.mp4','')}",
            metadata_rows=metadata_rows,
            label='fake'
        )

        fake_count += len(timestamps)

    # save metadata..

    metadata_file = os.path.join(OUTPUT_ROOT, 'metadata.csv')

    with open(
        metadata_file,
        'w',
        newline=''
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            'label',
            'video',
            'frame_index',
            'filename'
        ])

        writer.writerows(metadata_rows)

    # summary...

    print(f'\nreal frames : {real_count}')
    print('fake frames : {fake_count}')

if __name__ == '__main__':
    main()
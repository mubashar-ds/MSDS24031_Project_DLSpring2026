import os
import csv
import random
import cv2
import numpy as np
from tqdm import tqdm

# config...

DATASET_ROOT = '../datasets/video_faceforensics++_c23'
OUTPUT_ROOT = '../datasets/ffpp_frames'

FAKE_METHODS = [
    'Deepfakes',
    'Face2Face',
    'FaceSwap',
    'FaceShifter',
    'NeuralTextures'
]

TRAIN_COUNT = 300
VAL_COUNT = 50
TEST_COUNT = 50

FRAMES_REAL = 25
FRAMES_PER_METHOD = 5

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# helpers...

def get_all_real_ids():

    real_path = os.path.join(DATASET_ROOT, 'original')

    files = [
        f for f in os.listdir(real_path)
        if f.endswith('.mp4')
    ]

    return sorted([f.split('.')[0] for f in files])

def filter_valid_ids(all_ids):

    print('\nfiltering valid ids...')

    valid_ids = []

    for vid in tqdm(all_ids):
        valid = True

        for method in FAKE_METHODS:
            method_path = os.path.join(DATASET_ROOT, method)

            exists = any(f.startswith(vid + '_') for f in os.listdir(method_path))

            if not exists:
                valid = False
                break

        if valid:
            valid_ids.append(vid)

    return valid_ids

def split_ids(all_ids):
    random.shuffle(all_ids)

    train_ids = all_ids[:TRAIN_COUNT]
    val_ids = all_ids[TRAIN_COUNT: TRAIN_COUNT + VAL_COUNT]
    test_ids = all_ids[TRAIN_COUNT + VAL_COUNT: TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

    return train_ids, val_ids, test_ids

def get_fake_counterpart(method_path, source_id):

    candidates = sorted([
        f for f in os.listdir(method_path)
        if f.endswith('.mp4')
        and f.startswith(source_id + '_')
    ])

    if len(candidates) == 0:
        return None

    # deterministic choice..
    return candidates[0]

# timestamp generation..

def generate_diverse_indices(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if total <= 0:
        return None

    segments = np.linspace(0, total, num_frames + 1, dtype=int)

    indices = []

    for i in range(num_frames):
        start = segments[i]
        end = segments[i + 1]

        if start < end:
            idx = random.randint(start, end - 1)
            indices.append(idx)

    return sorted(indices)

# frame extraction..

def extract_specific_frames(video_path, output_dir,frame_indices,prefix,metadata_rows,split_name,label,method_name):

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    target_set = set(frame_indices)

    current = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if current in target_set:
            local_idx = frame_indices.index(current)

            filename = (f'{prefix}_{local_idx:02d}.png')

            save_path = os.path.join(output_dir, filename)

            cv2.imwrite(save_path, frame)

            metadata_rows.append([
                split_name,
                label,
                method_name,
                prefix,
                current,
                filename
            ])

        current += 1

    cap.release()

# process split..

def process_split(split_name, ids, metadata_rows):

    print(f'\nprocessing {split_name}')

    for vid in tqdm(ids):

        # real video..

        real_video = os.path.join(DATASET_ROOT, 'original', vid + '.mp4')
        timestamps = generate_diverse_indices(real_video, FRAMES_REAL)

        if timestamps is None:
            continue

        # real frames..

        real_out = os.path.join(OUTPUT_ROOT, split_name, 'real')

        extract_specific_frames(
            real_video,
            real_out,
            timestamps,
            prefix=f'real_{vid}',
            metadata_rows=metadata_rows,
            split_name=split_name,
            label='real',
            method_name='original'
        )

        # shuffle timestamps..

        shuffled = timestamps.copy()
        random.shuffle(shuffled)

        method_timestamp_map = {}

        for i, method in enumerate(FAKE_METHODS):

            start = i * FRAMES_PER_METHOD
            end = start + FRAMES_PER_METHOD

            method_timestamp_map[method] = sorted(shuffled[start:end])

        # fake methods..

        for method in FAKE_METHODS:

            method_path = os.path.join(DATASET_ROOT, method)

            fake_video_name = get_fake_counterpart(method_path, vid)

            if fake_video_name is None:
                continue

            fake_video = os.path.join(method_path,fake_video_name)

            fake_out = os.path.join(OUTPUT_ROOT, split_name, 'fake')

            extract_specific_frames(
                fake_video,
                fake_out,
                method_timestamp_map[method],
                prefix=f"{method}_{fake_video_name.split('.')[0]}",
                metadata_rows=metadata_rows,
                split_name=split_name,
                label="fake",
                method_name=method
            )

# main...

def main():

    all_ids = get_all_real_ids()

    print(f'\ntotal real videos: {len(all_ids)}')

    valid_ids = filter_valid_ids(all_ids)

    print(f'\nvalid IDs: {len(valid_ids)}')

    train_ids, val_ids, test_ids = split_ids(valid_ids)

    print('\ndataset split')
    print('train:', len(train_ids))
    print('val:', len(val_ids))
    print('test:', len(test_ids))

    metadata_rows = []

    process_split('train', train_ids, metadata_rows)

    process_split('val',val_ids, metadata_rows)

    process_split('test', test_ids, metadata_rows)

    # save metadata..

    metadata_file = os.path.join(OUTPUT_ROOT, 'metadata.csv')

    with open(metadata_file, 'w', newline='') as f:

        writer = csv.writer(f)

        writer.writerow([
            'split',
            'label',
            'method',
            'video_id',
            'frame_index',
            'filename'
        ])

        writer.writerows(metadata_rows)

    print('\ndone')
    print(f'metadata saved to: {metadata_file}')

if __name__ == '__main__':
    main()
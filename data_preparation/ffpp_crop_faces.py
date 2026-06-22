import os
import csv
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

# config...

INPUT_ROOT = '../datasets/ffpp_frames'
OUTPUT_ROOT = '../datasets/ffpp_cropped_faces'

IMG_SIZE = 384
MARGIN = 0.40

MIN_FACE_AREA_RATIO = 0.00

# retinaface..

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])

app.prepare(ctx_id=-1, det_size=(640, 640))

print('\nretinaface initialized')

# get image paths..

def get_all_images(root):

    data = []

    for split in ['train', 'val', 'test']:
        for cls in ['real', 'fake']:
            folder = os.path.join(root, split, cls)

            if not os.path.exists(folder):
                continue

            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):

                    data.append((os.path.join(folder, f), split, cls))

    return data

# face crop..

def crop_face_retina(img):

    faces = app.get(img)

    if len(faces) == 0:
        return None, None

    # largest face..

    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    x1, y1, x2, y2 = face.bbox.astype(int)

    h, w, _ = img.shape

    bw = x2 - x1
    bh = y2 - y1

    face_area = bw * bh
    image_area = h * w

    # reject tiny detections..

    if face_area < MIN_FACE_AREA_RATIO * image_area:
        return None, None

    # margin..

    x1 = int(max(0, x1 - MARGIN * bw))
    y1 = int(max(0, y1 - MARGIN * bh))

    x2 = int(min(w, x2 + MARGIN * bw))
    y2 = int(min(h, y2 + MARGIN * bh))

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    bbox = [x1, y1, x2, y2]

    return crop, bbox

# main..

def process_all():

    data = get_all_images(INPUT_ROOT)

    print(f'\ntotal images found: {len(data)}')

    saved = 0
    skipped = 0

    failed_files = []
    metadata_rows = []

    for path, split, cls in tqdm(data):
        img = cv2.imread(path)

        if img is None:
            skipped += 1
            failed_files.append(path)
            continue

        face, bbox = crop_face_retina(img)

        if face is None:
            skipped += 1
            failed_files.append(path)
            continue

        # resize..

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        save_dir = os.path.join(OUTPUT_ROOT, split, cls)

        os.makedirs(save_dir, exist_ok=True)

        original_name = os.path.basename(path)

        filename = (os.path.splitext(original_name)[0] + '.png')

        save_path = os.path.join(save_dir, filename)

        # save as png.

        cv2.imwrite(save_path, face)

        metadata_rows.append([
            filename,
            split,
            cls,
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3]
        ])

        saved += 1

    # save failed files..

    failed_file_path = os.path.join(OUTPUT_ROOT, 'failed_faces.txt')

    with open(failed_file_path, 'w') as f:

        for item in failed_files:
            f.write(item + '\n')

    # save metadata..

    metadata_file = os.path.join(OUTPUT_ROOT, 'metadata.csv')

    with open(metadata_file, 'w', newline='') as f:

        writer = csv.writer(f)

        writer.writerow([
            'filename',
            'split',
            'label',
            'x1',
            'y1',
            'x2',
            'y2'
        ])

        writer.writerows(metadata_rows)

    # summary..

    print('\nsaved :', saved)
    print('skipped :', skipped)

    print(f'\nmetadata : {metadata_file}')
    print(f'failed list : {failed_file_path}')

if __name__ == '__main__':
    process_all()
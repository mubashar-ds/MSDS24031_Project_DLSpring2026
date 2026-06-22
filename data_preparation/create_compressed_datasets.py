import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# configuration...

FFPP_TEST_DIR = '../datasets/ffpp_cropped_faces/test'
CELEBV2_DIR = '../datasets/celebv2_cropped_faces'

OUTPUT_ROOT = '../datasets/compressed_ffpp_celebv2'

JPEG_QUALITIES = [90, 70, 50, 30, 10]

# compression function..

def compress_dataset(source_dir, dataset_name, qualities=JPEG_QUALITIES,):
    
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f'dataset not found: {source_dir}'
        )

    print(f'\nprocessing dataset: {dataset_name}')

    for quality in qualities:

        print(f'\ncreating JPEG Q{quality} dataset...')

        output_dataset_dir = (
            Path(OUTPUT_ROOT)
            / dataset_name
            / f'q{quality}'
        )

        total_images_processed = 0

        for class_name in ['real', 'fake']:

            src_class_dir = source_dir / class_name
            dst_class_dir = output_dataset_dir / class_name

            dst_class_dir.mkdir(parents=True, exist_ok=True)

            image_paths = list(src_class_dir.rglob('*.png'))

            print(
                f'{class_name}: '
                f'{len(image_paths)} images'
            )

            for img_path in tqdm(image_paths, desc=f'Q{quality} {class_name}',):

                try:
                    img = Image.open(img_path).convert('RGB')

                    output_path = (
                        dst_class_dir
                        / f'{img_path.stem}.jpg'
                    )

                    img.save(
                        output_path,
                        format='JPEG',
                        quality=quality,
                        optimize=True,
                        subsampling=0,
                    )

                    total_images_processed += 1

                except Exception as e:

                    print(f'\nfailed: {img_path}\n{e}')

        print(
            f'\nCompleted {dataset_name} '
            f'Q{quality} '
            f'({total_images_processed} images)'
        )

    print(f'\nfinished dataset: {dataset_name}')

# verify counts..

def verify_counts():

    print('\nverifying image counts...')

    datasets = [('ffpp', FFPP_TEST_DIR), ('celebv2', CELEBV2_DIR),]

    for dataset_name, source_dir in datasets:

        print('\n')    
        print(dataset_name.upper())

        source_dir = Path(source_dir)

        for cls in ['real', 'fake']:

            original_count = len(
                list(
                    (source_dir / cls)
                    .rglob('*.png')
                )
            )

            print(
                f'\noriginal {cls}: '
                f'{original_count}'
            )

            for q in JPEG_QUALITIES:

                compressed_count = len(
                    list(
                        (
                            Path(OUTPUT_ROOT)
                            / dataset_name
                            / f'q{q}'
                            / cls
                        ).rglob('*.jpg')
                    )
                )

                status = (
                    'correct...'
                    if compressed_count == original_count
                    else 'incorrect!'
                )

                print(
                    f'\nQ{q}: '
                    f'{compressed_count} '
                    f'{status}'
                )


# main...

if __name__ == '__main__':

    compress_dataset(FFPP_TEST_DIR, 'ffpp')
    compress_dataset(CELEBV2_DIR, 'celebv2')

    verify_counts()

    print('\nall datasets generated successfully...')

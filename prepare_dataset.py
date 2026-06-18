#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def opts() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data-file-path',
                        help='The MIN JSON data file path',
                        type=str,
                        required=True)
    parser.add_argument(
        '-c',
        '--classes',
        help='list of the dataset classes (e.g., adult,juvenile)',
        required=True)
    parser.add_argument('-b',
                        '--classify-by',
                        help='Dict key to classify the dataset by',
                        required=True)
    parser.add_argument('-o',
                        '--output-dir',
                        help='Output path',
                        required=True)
    parser.add_argument('-s',
                        '--split_ratio',
                        help='Train-val split; train split as a float',
                        type=float,
                        default=0.8)
    return parser.parse_args()


def crop_and_save_image(image, bbox, save_path):
    """
    Crop and save an image based on the bounding box.

    Args:
        image (str): The image path.
        bbox (dict): The bounding box.
        save_path (str): The path to save the cropped image.
    """
    x_min = int(bbox['x'] / 100 * bbox['original_width'])
    y_min = int(bbox['y'] / 100 * bbox['original_height'])
    x_max = int((bbox['x'] + bbox['width']) / 100 * bbox['original_width'])
    y_max = int((bbox['y'] + bbox['height']) / 100 * bbox['original_height'])

    image = Image.open(image)
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image.save(save_path)


def main():
    """
    Main function to prepare the classification dataset.

    The function reads the MIN JSON data file, crops the images based on the
    bounding boxes, and saves the cropped images in the respective class
    folders.
    """
    args = opts()

    data_file = args.data_file_path
    dataset_path = args.output_dir

    Path(dataset_path).mkdir(exist_ok=True)

    classes = [x.strip() for x in args.classes.split(',')]

    with open(data_file) as j:
        data = json.load(j)

    proj_name = args.output_dir

    all_images = glob.glob('uploaded_label_studio_images/*/*')
    images_dict = {Path(x).name: x for x in all_images}

    exclude_images = [Path(x).stem for x in glob.glob('raccoon/*/*')]

    for c in classes:
        Path(f'{proj_name}/{c}').mkdir(exist_ok=True, parents=True)

    for d in tqdm(data):
        if not d.get(args.classify_by):
            continue

        if not d.get('rect-1'):
            continue

        image = d['image']
        if Path(image).stem in exclude_images:
            continue
        try:
            image_relative_path = images_dict[Path(image).name]
        except KeyError as e:
            print(f'Skipped {image}', e)
            continue
        #image_relative_path = f'images/{Path(image).name}'

        _cls = d.get(args.classify_by)
        if isinstance(_cls, str):
            _cls = [_cls]

        # Label Studio choices can sometimes be lists of lists, e.g. [['adult'], ['juvenile']]
        _cls = [c[0] if isinstance(c, list) else c for c in _cls]

        # If there's only one class but multiple bounding boxes, broadcast the class
        if len(_cls) == 1 and len(d['rect-1']) > 1:
            _cls = _cls * len(d['rect-1'])
        elif len(_cls) != len(d['rect-1']):
            print(f"Warning: length mismatch in {image}. Rects: {len(d['rect-1'])}, Classes: {len(_cls)}")

        if not Path(image_relative_path).exists():
            print(f'Image does not exist! {image}')
            continue

        for n, (bbox, c) in enumerate(zip(d['rect-1'], _cls)):
            if c in ['unsure', 'exclude']:
                continue
            img_name = f'{Path(image).stem}.{n}{Path(image).suffix}'
            save_path = f'{proj_name}/{c}/{img_name}'
            crop_and_save_image(image_relative_path, bbox, save_path)

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    for class_name in classes:
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)

    for class_name in classes:
        images = glob.glob(os.path.join(dataset_path, class_name, '*'))
        random.shuffle(images)
        split_index = int(args.split_ratio * len(images))
        train_images = images[:split_index]
        val_images = images[split_index:]

        for img_path in tqdm(train_images):
            shutil.copy2(img_path, os.path.join(train_path, class_name))

        for img_path in tqdm(val_images):
            shutil.copy2(img_path, os.path.join(val_path, class_name))

        shutil.rmtree(os.path.join(dataset_path, class_name))


if __name__ == "__main__":
    main()

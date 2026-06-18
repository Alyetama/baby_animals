#!/usr/bin/env python
# coding: utf-8

import argparse
import re
from pathlib import Path
from urllib import request

from tqdm import tqdm


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--html-file',
                        help='HTML body file',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output-dir',
                        help='Output directory for the downloaded images',
                        type=str,
                        default='downloaded_images')
    return parser.parse_args()


def get_and_download(html_file, output_dir):
    with open(html_file) as f:
        html = f.read()

    pattern = re.compile(r'(?P<url>/photos/[^"\']*/deployment/[^"\']+)',
                         re.IGNORECASE)
    matches = [m.group('url') for m in pattern.finditer(html)]
    matches = list(set(matches))
    print(f'Downloading {len(matches)} images...')

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for match in tqdm(matches):
        url = f'http://inquire-snapshot-usa.csail.mit.edu{match}'
        request.urlretrieve(url, f'{output_dir}/{Path(url).name}')


def main():
    args = opts()
    get_and_download(html_file=args.html_file, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

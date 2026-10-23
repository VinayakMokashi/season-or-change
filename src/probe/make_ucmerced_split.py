"""Write a stratified 80/20 UC Merced split (80 train / 20 test images per class)."""
import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='folder with one sub-folder per class')
    parser.add_argument('--out_dir', default='splits')
    parser.add_argument('--test_per_class', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    train, test = [], []
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        files = sorted(f'{cls_dir.name}/{p.name}' for p in cls_dir.iterdir() if p.is_file())
        rng.shuffle(files)
        test += sorted(files[:args.test_per_class])
        train += sorted(files[args.test_per_class:])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ucmerced_train.txt').write_text('\n'.join(train) + '\n', encoding='utf-8', newline='\n')
    (out / 'ucmerced_test.txt').write_text('\n'.join(test) + '\n', encoding='utf-8', newline='\n')
    print(f'train={len(train)} test={len(test)} -> {out}')


if __name__ == '__main__':
    main()

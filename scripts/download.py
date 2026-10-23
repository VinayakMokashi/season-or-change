"""Download evaluation data and encoder weights.

  python scripts/download.py            # EuroSAT, UC Merced, all encoder weights
  python scripts/download.py --pretrain # also the 2k-location pretraining archive (1.5 GB)

Layout created (all git-ignored):
  data/eurosat/EuroSAT_RGB/{<class>/*.jpg, train.txt, val.txt}
  data/ucmerced/UCMerced_LandUse/Images/<class>/*.tif
  checkpoints/ours_10k/resnet18_{seco,teco,caco}_geo_10k_400.pth
  checkpoints/authors_100k/resnet18_{seco,teco,caco}_geo_100k_1000.pth, resnet18_moco_geo_100k_200.pth
  data/clean_10k_geography/                      (only with --pretrain)

Our encoders come from this repository's v1.0 GitHub release, with Google Drive
as a fallback; the CACo authors' encoders and the datasets come from their
original hosts.

Files are written to a temporary ".part" file and renamed when complete, so an
interrupted download is retried on the next run. Downloaded encoder weights are
checked against their SHA-256 hashes; existing files are left untouched.
"""
import argparse
import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

CACO = 'https://research.cs.cornell.edu/caco'
UCM_URL = 'https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip'
RELEASE = 'https://github.com/VinayakMokashi/season-or-change/releases/download/v1.0/{}'
DRIVE = 'https://drive.usercontent.google.com/download?id={}&export=download&confirm=t'

# Our encoders (GitHub release, Google Drive as a mirror): file -> (Drive id, sha256)
OURS = {
    'resnet18_seco_geo_10k_400.pth': ('13pSq2P2S1n1CfpIo35g_oRwCHrAiXLsI',
                                      'e19e1d4aa89f0407a197e25d79580ea79a4a817c15652edbc06503d7af91d11d'),
    'resnet18_teco_geo_10k_400.pth': ('1FhsYCjCH8x7fI5pEri4RjCgACfrE7ttj',
                                      '9414c6fb35ac97ac6ba74a5b19358e6bbaba10325e0a0c0db4cd6a1d470f7f01'),
    'resnet18_caco_geo_10k_400.pth': ('1tVNnM9UqlvAUuBAmjdwzWFVvtrzV67Zx',
                                      'c99978cb1359d1fca54660bcf56148bfe0ae0ffddfb2cc90a79ba9246611acef'),
}
# The CACo authors' released encoders: file -> sha256
AUTHORS = {
    'resnet18_seco_geo_100k_1000.pth': '882d999d00436447138533cfe088fc9f9ff777f6abe82c7767af0883241b35de',
    'resnet18_teco_geo_100k_1000.pth': 'a3d0e1771b7b3b05cc4938bec37368c60c86cb31e3512c87f52aa7785260fc76',
    'resnet18_caco_geo_100k_1000.pth': '92ab3a457958f28f666209450feec10d0c43b0949ed02a8faa01e6ca0dc45d66',
    'resnet18_moco_geo_100k_200.pth': 'db3616d87f68659c830789d69fb2b0717f33042412eaeb3f44c0393d253e6375',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(urls, dst, expected_sha256=None):
    """Download the first working URL in `urls` (a string or a list) to `dst`."""
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + '.part')
    for url in [urls] if isinstance(urls, str) else urls:
        print(f'downloading {dst.name}')
        try:
            with urllib.request.urlopen(url) as r, open(part, 'wb') as f:
                shutil.copyfileobj(r, f)
        except OSError as err:
            print(f'  {url} failed ({err}); trying the next source')
            continue
        if expected_sha256 and sha256(part) != expected_sha256:
            print('  SHA-256 mismatch; trying the next source')
            continue
        part.replace(dst)
        return dst
    part.unlink(missing_ok=True)
    raise RuntimeError(f'could not download {dst.name}; run the script again')


def extract(archive, target, marker):
    """Extract into `target` unless `marker` exists; mark completion with a sentinel file."""
    done = Path(target) / f'.{Path(archive).name}.extracted'
    if done.exists() and Path(marker).exists():
        return
    if str(archive).endswith('.zip'):
        zipfile.ZipFile(archive).extractall(target)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(target)
    done.touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pretrain', action='store_true', help='also download the pretraining archive')
    args = parser.parse_args()

    for name, (file_id, digest) in OURS.items():
        fetch([RELEASE.format(name), DRIVE.format(file_id)], f'checkpoints/ours_10k/{name}', digest)
    for name, digest in AUTHORS.items():
        fetch(f'{CACO}/checkpoints/{name}', f'checkpoints/authors_100k/{name}', digest)

    eurosat = Path('data/eurosat')
    z = fetch(f'{CACO}/data/eurosat/EuroSAT_RGB.zip', eurosat / 'EuroSAT_RGB.zip')
    extract(z, eurosat, eurosat / 'EuroSAT_RGB')
    for split in ('train', 'val'):
        fetch(f'{CACO}/data/eurosat/{split}.txt', eurosat / 'EuroSAT_RGB' / f'{split}.txt')

    ucm = Path('data/ucmerced')
    z = fetch(UCM_URL, ucm / 'UCMerced_LandUse.zip')
    extract(z, ucm, ucm / 'UCMerced_LandUse')

    if args.pretrain:
        t = fetch(f'{CACO}/data/caco/caco10k.tar.gz', 'data/caco10k.tar.gz')
        extract(t, 'data', 'data/clean_10k_geography')
    print('done')


if __name__ == '__main__':
    main()

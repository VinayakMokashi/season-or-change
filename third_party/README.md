# Third-party code

`caco_upstream/` is an unmodified copy of the CACo reference implementation
by Utkarsh Mall, Bharath Hariharan and Kavita Bala:

- Source: <https://github.com/utkarshmall13/CACo>
- Commit: `7c4fbc8d9e65a4ef715596c5c275ab3ddb6a8193` (2024-04-03)
- License: MIT, Copyright (c) 2023 Utkarsh Mall (see `caco_upstream/LICENSE`)

Our pretraining wrapper (`src/pretrain/train.py`) imports its models, datasets
and training callbacks without changes.

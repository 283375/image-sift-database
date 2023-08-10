# image-sift-database

Store your images SIFT descriptors using JSON/sqlite3.

## Usage

See [build.py](./build.py) for minimal usage:

```py
from build_database import SIFTDatabase

SIFTDatabase().build((100, 100))
```

It

1. creates a time-named database file
2. `CREATE TABLE sift (id TEXT PRIMARY KEY, descriptors BLOB)`
3. reads all the image files (ends with extension `.jpg`/`.png`) under `images/`
4. resizes all the images to `100 * 100`
5. calculates the image descriptors
6. `INSERT INTO sift (id, descriptors) VALUES (?, ?)`, `[image_path.stem, descriptors.tobytes()]`
7. `conn.commit()`

## API

```py
from pathlib import Path

class SIFTDatabase:
    EXTS = [".jpg", ".png"]

    def __init__(
        self,
        images_path: Path = Path(os.getcwd()) / "images",
        output_file: str = f"sift_{int(time.time())}.db",
    ): ...
```

See [build_database.py](./build_database.py) for details.

## Notes

This repository's `.gitignore` is an "allow list":

```
**/*

!.git*
!/build_database.py
!/build.py
!/README.md
```

so you can safely put any files you need right after you clone this repository, and easily update the script with `git pull`. Have fun using!

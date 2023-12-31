import os
import sqlite3
import time
import traceback
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from tqdm import tqdm

# from pylance
Mat = np.ndarray[int, np.dtype[np.generic]]


class SIFTDatabase:
    EXTS = [".jpg", ".png"]

    def __init__(
        self,
        images_path: Path = Path(os.getcwd()) / "images",
        output_file: str = f"sift_{int(time.time())}.db",
        *,
        filename_hook: Optional[Callable[[Path], str]] = None,
        img_hook: Optional[Callable[[Mat], Mat]] = None,
        gzip: int = 0,
    ):
        self.__images_path = images_path
        self.__output_file = output_file
        self.__sift = cv2.SIFT_create()

        self.__filename_hook = filename_hook or (lambda p: p.stem)
        self.__img_hook = img_hook
        self.__gzip = gzip

    @property
    def images_path(self):
        return self.__images_path

    @images_path.setter
    def images_path(self, value: Path):
        self.__images_path = value

    @property
    def output_file(self):
        return self.__output_file

    @output_file.setter
    def output_file(self, value: str):
        self.__output_file = value

    @property
    def sift(self):
        return self.__sift

    @property
    def filename_hook(self):
        return self.__filename_hook

    @filename_hook.setter
    def filename_hook(self, value: Callable[[Path], str]):
        self.__filename_hook = value

    @property
    def img_hook(self):
        return self.__img_hook

    @img_hook.setter
    def img_hook(self, value: Callable[[Mat], Mat]):
        self.__img_hook = value

    @property
    def gzip(self):
        return self.__gzip

    @gzip.setter
    def gzip(self, value: int):
        self.__gzip = value

    def sift_detectAndCompute(self, __img, SZ: tuple[int, int] | None = None):
        img = __img.copy()
        if SZ is not None:
            img = cv2.resize(img, SZ)
        return self.sift.detectAndCompute(img, None)

    def build(self, SZ: tuple[int, int] | None = None):
        # sourcery skip: extract-method
        images = [
            p for p in self.images_path.glob("**/*") if p.suffix.lower() in self.EXTS
        ]

        conn = sqlite3.connect(self.output_file)
        conn.execute("PRAGMA journal_mode=WAL;")
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE sift (id INTEGER PRIMARY KEY, tag TEXT, descriptors BLOB)"
            )
            cursor.execute("CREATE TABLE properties (id TEXT UNIQUE, value TEXT)")
            insert_batch: list[tuple[str, bytes]] = []
            for image_path in tqdm(images):
                try:
                    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if self.img_hook:
                        img = self.img_hook(img)
                    _, descriptors = self.sift_detectAndCompute(img, SZ)
                    buffer = BytesIO()
                    if self.gzip:
                        gzipped_protocol_buffer = GzipFile(None, "wb", fileobj=buffer)
                        np.save(gzipped_protocol_buffer, descriptors)
                        gzipped_protocol_buffer.close()
                    else:
                        np.save(buffer, descriptors)
                    buffer.seek(0)
                    insert_batch.append((self.filename_hook(image_path), buffer.read()))
                except Exception as e:
                    tqdm.write("".join(traceback.format_exception(e)))
            print(f"Saving to {self.output_file}")
            cursor.executemany(
                "INSERT INTO sift (tag, descriptors) VALUES (?, ?)", insert_batch
            )
            cursor.executemany(
                "INSERT INTO properties (id, value) VALUES (?, ?)",
                [
                    ("size", f"{SZ[0]}, {SZ[1]}"),
                    ("gzip", self.gzip > 0),
                    ("build_date", int(time.time() * 1000)),
                ],
            )
            conn.commit()

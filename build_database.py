import base64
import json
import os
from pathlib import Path
import sqlite3
import traceback
import time

import numpy as np
import cv2
from tqdm import tqdm


class SIFTDatabase:
    EXTS = [".jpg", ".png"]

    def __init__(
        self,
        images_path: Path = Path(os.getcwd()) / "images",
        output_file: str = f"sift_{int(time.time())}.db",
    ):
        self.__images_path = images_path
        self.__output_file = output_file
        self.__sift = cv2.SIFT_create()

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

    def sift_detectAndCompute(self, __img_gray, SZ: tuple[int, int] | None = None):
        img_gray = __img_gray.copy()
        if SZ is not None:
            img_gray = cv2.resize(img_gray, SZ)
        return self.sift.detectAndCompute(img_gray, None)

    def build(self, SZ: tuple[int, int] | None = None):
        images = [p for p in self.images_path.glob("**/*") if p.suffix in self.EXTS]

        if self.output_file.endswith(".json"):
            output_dict = {}
            for image_path in tqdm(images):
                try:
                    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    _, descriptors = self.sift_detectAndCompute(img_gray, SZ)
                    output_dict[image_path.stem] = base64.b64encode(
                        descriptors.tobytes()
                    ).decode("utf-8")
                except Exception as e:
                    tqdm.write("".join(traceback.format_exception(e)))

            with open(self.output_file, "w", encoding="utf-8") as output_f:
                json.dump(output_dict, output_f, ensure_ascii=False)
        elif self.output_file.endswith(".db"):
            conn = sqlite3.connect(self.output_file)
            conn.execute("PRAGMA journal_mode=WAL;")
            with conn:
                conn.execute(
                    "CREATE TABLE sift (id TEXT PRIMARY KEY, descriptors BLOB)"
                )
                cursor = conn.cursor()
                insert_batch: list[tuple[str, bytes]] = []
                for image_path in tqdm(images):
                    try:
                        img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        _, descriptors = self.sift_detectAndCompute(img_gray, SZ)
                        insert_batch.append((image_path.stem, descriptors.tobytes()))
                    except Exception as e:
                        tqdm.write("".join(traceback.format_exception(e)))
                cursor.executemany(
                    'INSERT INTO sift (id, descriptors) VALUES (?, ?)', insert_batch
                )
                conn.commit()

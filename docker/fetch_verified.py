"""fetch_verified.py URL MD5 DEST [LOCAL]

Put at DEST a file with the given MD5: the LOCAL copy if it exists and matches,
otherwise a download from URL, refused if its MD5 differs. Used in the build
stage of the graph images, so the graphs are fetched from their publishers and
never redistributed. Standard library only, so an image builds offline when
every file is supplied locally.
"""

import hashlib
import os
import shutil
import sys
import time
import urllib.request

ATTEMPTS = 5


def md5_of(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "nexa-build/1.0"})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            return
        except OSError as error:
            if attempt == ATTEMPTS:
                raise
            print(f"  attempt {attempt} failed ({error}); retrying", flush=True)
            time.sleep(5 * attempt)


def main(url: str, md5: str, dest: str, local_copy: str = "") -> None:
    if local_copy and os.path.isfile(local_copy):
        if md5_of(local_copy) == md5:
            print(f"using the local copy {local_copy}", flush=True)
            shutil.copyfile(local_copy, dest)
            return
        print(f"the local copy {local_copy} does not match MD5 {md5}: downloading instead", flush=True)
    print(f"downloading {url}", flush=True)
    download(url, dest + ".part")
    got = md5_of(dest + ".part")
    if got != md5:
        sys.exit(f"MD5 mismatch for {url}: expected {md5}, got {got}")
    os.replace(dest + ".part", dest)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__.splitlines()[0])
    main(*sys.argv[1:])

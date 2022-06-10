"""Download facebook flore dataset."""
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Union

import portalocker

# global logger for this project
logger = logging.getLogger("lanmttrainer")


def count_lines(file_path: Union[str, Path]):
    """Count total lines of given file."""
    with open(file_path, "r") as f:
        return sum(1 for _ in f)


def check_md5sum(dest_path, expected_md5):
    """Check md5sum of a file."""
    md5 = hashlib.md5()
    with open(dest_path, "rb") as infile:
        for line in infile:
            md5.update(line)
        cur_md5 = md5.hexdigest()
    if cur_md5 != expected_md5:
        # do somethings
        pass
    else:
        # do somethings
        pass


def download_file(source_path, dest_path, extract_to=None, expected_md5=None):
    """Util function to download file from source_path to dest_path.

    :param source_path: the remote uri to download
    :param dest_path: where to save the file
    :param extract_to: for tarballs, where to extract to
    :param expected_md5: the MD5 sum
    :return: the set of processed file names
    """
    import ssl
    import urllib.request

    outdir = os.path.dirname(dest_path)
    os.makedirs(outdir, exist_ok=True)

    lockfile = os.path.join(outdir, f"{os.path.basename(dest_path)}.lock")
    with portalocker.Lock(lockfile, "w", timeout=60):
        if os.path.exists(dest_path):
            check_md5sum(dest_path, expected_md5)
        else:
            try:
                with urllib.request.urlopen(source_path) as f, open(
                    dest_path, "wb"
                ) as out:
                    out.write(f.read())
            except ssl.SSLError:
                logger.warning(
                    "An SSL error was encountered in downloading the files. If you're on a Mac, "
                    'you may need to run the "Install Certificates.command" file located in the '
                    '"Python 3" folder, often found under /Applications'
                )
                sys.exit(1)

            check_md5sum(dest_path, expected_md5)

            # Extract the tarball
            if extract_to is not None:
                shutil.unpack_archive(dest_path, extract_to)

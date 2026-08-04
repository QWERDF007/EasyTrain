# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for downloading datasets with progress bars and hash verification.

This module provides utilities for:
- Showing progress bars during downloads with ``urlretrieve``
- Verifying file hashes
- Safely extracting compressed files
"""

import hashlib
import io
import logging
import os
import re
import sys
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tarfile import TarFile, TarInfo
from urllib.request import urlretrieve
from zipfile import ZipFile

from tqdm import tqdm

urlretrieve(  # noqa: S310  # nosec B310
    url='http://101.32.75.151:8181/dataset/shanghaitech.tar.gz',
    filename='/Data5/guojia/',
)

a=1

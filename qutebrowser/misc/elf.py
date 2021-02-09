# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:

# Copyright 2021 Florian Bruhin (The-Compiler) <mail@qutebrowser.org>
#
# This file is part of qutebrowser.
#
# qutebrowser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# qutebrowser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with qutebrowser.  If not, see <https://www.gnu.org/licenses/>.

"""Simplistic ELF parser to get the QtWebEngine/Chromium versions.

I know what you must be thinking when reading this: "Why on earth does qutebrowser has
an ELF parser?!". For one, because writing one was an interesting learning exercise. But
there's actually a reason it's here: QtWebEngine 5.15.x versions come with different
underlying Chromium versions, but there is no API to get the version of
QtWebEngine/Chromium...

We can instead:

a) Look at the Qt runtime version (qVersion()). This often doesn't actually correspond
to the QtWebEngine version (as that can be older/newer). Since there will be a
QtWebEngine 5.15.3 release, but not Qt itself (due to LTS licensing restrictions), this
isn't a reliable source of information.

b) Look at the PyQtWebEngine version (PyQt5.QtWebEngine.PYQT_WEBENGINE_VERSION_STR).
This is a good first guess (especially for our Windows/macOS releases), but still isn't
certain. Linux distributions often push a newer QtWebEngine before the corresponding
PyQtWebEngine release, and some (*cough* Gentoo *cough*) even publish QtWebEngine
"5.15.2" but upgrade the underlying Chromium.

c) Parse the user agent. This is what qutebrowser did before this monstrosity was
introduced (and still does as a fallback), but for some things (finding the proper
commandline arguments to pass) it's too late in the initialization process.

d) Spawn QtWebEngine in a subprocess and ask for its user-agent. This takes too long to
do it on every startup.

e) Ask the package manager for this information. This means we'd need to know (or guess)
the package manager and package name. Also see:
https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=752114

Because of all those issues, we instead look for the (fixed!) version string as part of
the user agent header. Because libQt5WebEngineCore is rather big (~120 MB), we don't
want to search through the entire file, so we instead have a simplistic ELF parser here
to find the .rodata section. This way, searching the version gets faster by some orders
of magnitudes (a couple of us instead of ms).

This is a "best effort" parser. If it errors out, we instead end up relying on the
PyQtWebEngine version, which is the next best thing.
"""

import struct
import enum
import sys
import re
import dataclasses
import mmap
import pathlib
from typing import IO, ClassVar, Dict, Tuple, Optional

from PyQt5.QtCore import QLibraryInfo

from qutebrowser.utils import log


class ParseError(Exception):
    pass


class Bitness(enum.Enum):

    x32 = 1
    x64 = 2


class Endianness(enum.Enum):

    little = 1
    big = 2


def _unpack(fmt, fobj):
    size = struct.calcsize(fmt)

    try:
        data = fobj.read(size)
    except OSError as e:
        raise ParseError(e)

    try:
        return struct.unpack(fmt, data)
    except struct.error as e:
        raise ParseError(e)


@dataclasses.dataclass
class Ident:

    magic: bytes
    klass: Bitness
    data: Endianness
    version: int
    osabi: int
    abiversion: int

    _FORMAT: ClassVar[str] = '<4sBBBBB7x'
    # _SIZE: ClassVar[int] = 0x10

    @classmethod
    def parse(cls, fobj: IO[bytes]) -> 'Ident':
        magic, klass, data, version, osabi, abiversion = _unpack(cls._FORMAT, fobj)

        try:
            bitness = Bitness(klass)
        except ValueError:
            raise ParseError(f"Invalid bitness {klass}")

        try:
            endianness = Endianness(data)
        except ValueError:
            raise ParseError(f"Invalid endianness {data}")

        return cls(magic, bitness, endianness, version, osabi, abiversion)


@dataclasses.dataclass
class Header:

    typ: int
    machine: int
    version: int
    entry: int
    phoff: int
    shoff: int
    flags: int
    ehsize: int
    phentsize: int
    phnum: int
    shentsize: int
    shnum: int
    shstrndx: int

    _FORMATS: ClassVar[Dict[Bitness, str]] = {
        Bitness.x64: '<HHIQQQIHHHHHH',
        Bitness.x32: '<HHIIIIIHHHHHH',
    }

    # _SIZES: ClassVar[Dict[Bitness, int]] = {
    #     Bitness.x64: 0x30,
    #     Bitness.x32: 0x18,
    # }

    @classmethod
    def parse(cls, fobj: IO[bytes], bitness: Bitness) -> 'Header':
        fmt = cls._FORMATS[bitness]
        return cls(*_unpack(fmt, fobj))


@dataclasses.dataclass
class SectionHeader:

    name: int
    typ: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    addralign: int
    entsize: int

    _FORMATS: ClassVar[Dict[Bitness, str]] = {
        Bitness.x64: '<IIQQQQIIQQ',
        Bitness.x32: '<IIIIIIIIII',
    }

    # _SIZES: ClassVar[Dict[Bitness, int]] = {
    #     Bitness.x64: 0x40,
    #     Bitness.x32: 0x28,
    # }

    @classmethod
    def parse(cls, fobj: IO[bytes], bitness: Bitness) -> 'SectionHeader':
        fmt = cls._FORMATS[bitness]
        return cls(*_unpack(fmt, fobj))


def get_rodata_header(f: IO[bytes]) -> SectionHeader:
    ident = Ident.parse(f)
    if ident.magic != b'\x7fELF':
        raise ParseError(f"Invalid magic {ident.magic!r}")

    if ident.data != Endianness.little:
        raise ParseError("Big endian is unsupported")

    if ident.version != 1:
        raise ParseError(f"Only version 1 is supported, not {ident.version}")

    header = Header.parse(f, bitness=ident.klass)

    # Read string table
    f.seek(header.shoff + header.shstrndx * header.shentsize)
    shstr = SectionHeader.parse(f, bitness=ident.klass)

    f.seek(shstr.offset)
    string_table = f.read(shstr.size)

    # Back to all sections
    for i in range(header.shnum):
        f.seek(header.shoff + i * header.shentsize)
        sh = SectionHeader.parse(f, bitness=ident.klass)
        name = string_table[sh.name:].split(b'\x00')[0]
        if name == b'.rodata':
            return sh

    raise ParseError("No .rodata section found")


@dataclasses.dataclass
class Versions:

    webengine: str
    chromium: str


def _parse_from_path(path: pathlib.Path) -> Versions:
    with path.open('rb') as f:
        sh = get_rodata_header(f)

        try:
            rodata = mmap.mmap(
                f.fileno(),
                sh.size,
                offset=sh.offset,
                access=mmap.ACCESS_READ,
            )
        except OSError as e:
            raise ParseError(e)

        match = re.search(
            br'QtWebEngine/([0-9.]+) Chrome/([0-9.]+)',
            rodata,
        )  # type: ignore[call-overload]
        if match is None:
            raise ParseError("No match in .rodata")

        try:
            return Versions(
                webengine=match.group(1).decode('ascii'),
                chromium=match.group(2).decode('ascii'),
            )
        except UnicodeDecodeError as e:
            raise ParseError(e)


def parse_webenginecore() -> Optional[Versions]:
    library_path = pathlib.Path(QLibraryInfo.location(QLibraryInfo.LibrariesPath))
    lib_file = library_path / 'libQt5WebEngineCore.so'
    if not lib_file.exists():
        return None

    try:
        return _parse_from_path(lib_file)
    except ParseError as e:
        log.init.debug(f"Failed to parse ELF: {e}")
        return None

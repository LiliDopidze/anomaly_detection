"""Telecom sector pack v0.1."""

from importlib.resources import as_file, files

from telemetry_contract import DeclarativeSectorPack, load_declarative_pack

PACK_ID = "telecom"
PACK_VERSION = "0.1.0"


def load_pack() -> DeclarativeSectorPack:
    with as_file(files(__package__)) as pack_dir:
        return load_declarative_pack(pack_dir)


__all__ = ["PACK_ID", "PACK_VERSION", "load_pack"]

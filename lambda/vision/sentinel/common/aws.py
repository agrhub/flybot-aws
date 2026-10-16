"""boto3 session/client factory. Profile and region come from the environment, never from code.

    SENTINEL_AWS_PROFILE (or AWS_PROFILE)   CLI profile, e.g. `sentinel` on the dev laptop; unset in Lambda
    SENTINEL_AWS_REGION / AWS_REGION        default ap-southeast-2 (the account's SCP denies other regions)
"""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_REGION = "ap-southeast-2"


def region() -> str:
    return os.environ.get("SENTINEL_AWS_REGION") or os.environ.get("AWS_REGION") or DEFAULT_REGION


@lru_cache(maxsize=4)
def session(profile: str | None = None, region_name: str | None = None):
    import boto3

    profile = profile or os.environ.get("SENTINEL_AWS_PROFILE") or os.environ.get("AWS_PROFILE") or None
    return boto3.Session(profile_name=profile, region_name=region_name or region())


def client(service: str, **kwargs):
    return session().client(service, **kwargs)

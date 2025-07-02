"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2025-05-31
"""

from datetime import timezone, datetime

"""
Functions that record, interpret, and organize time.
"""

def now_as_dt_utc() -> datetime:
    """
    Return a datetime object for this moment (in UTC time).
    """
    return datetime.now(timezone.utc)

def now_as_millis_utc() -> float:
    """
    Return this moment measured in elapsed milliseconds (since the epoch start in utc timezone).
    """
    return dt_to_millis_utc(now_as_dt_utc())

def now_as_dt_local() -> datetime:
    """
    Returns a datetime object representing this moment (local timezone).
    """
    return datetime.now().astimezone()

def local_eod_as_dt_local() -> datetime:
    """
    Get the datetime object associated with the end of today (local) in local timezone.
    """
    return now_as_dt_local().replace(hour=23, minute=59, second=59, microsecond=999999)

def local_sod_as_dt_local() -> datetime:
    """
    Get the datetime object associated with the start of today (local) in local timezone.
    """
    return now_as_dt_local().replace(hour=0, minute=0, second=0, microsecond=0)

def dt_to_millis_utc(dt_obj: datetime) -> float:
    """
    Return the number of milliseconds that elapsed from the start of UTC epoch 
    (Jan 1, 1970 00:00:00 UTC) to now.
    """
    assert_has_timezone(dt_obj)
    start = datetime.fromtimestamp(0.0, timezone.utc)
    end = dt_to_dt_utc(dt_obj)
    return (end - start).total_seconds() * 1000.0

def millis_utc_to_dt_utc(millis: float) -> datetime:
    """
    Given a specification of elapsed milliseconds since UTC epoch, return a datetime object in
    utc timezone.
    """
    return datetime.fromtimestamp(millis / 1000.0, timezone.utc)

def millis_utc_to_dt_local(millis: float) -> datetime:
    """
    Given a specification of elapsed milliseconds since the UTC epoch, return a datetime object
    in local timezone that represents that moment.
    """
    return datetime.fromtimestamp(millis / 1000.0).astimezone()

def dt_to_dt_local(dt_obj: datetime) -> datetime:
    """
    Convert the given datetime object to local timezone. If already in local, does nothing.
    """
    assert_has_timezone(dt_obj)
    return dt_obj.astimezone()

def dt_to_dt_utc(dt_obj: datetime) -> datetime:
    """
    Convert the given datetime object to UTC timezone. If already in utc, does nothing.
    """
    assert_has_timezone(dt_obj)
    return dt_obj.astimezone(timezone.utc)

def assert_has_timezone(dt_obj: datetime) -> None:
    """
    Raises an error if the datetime object does not have a specified timezone.
    """
    if not dt_obj.tzinfo:
        raise ValueError("dt_obj does not have a timezone specification!")




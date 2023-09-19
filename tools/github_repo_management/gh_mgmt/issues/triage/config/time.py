from datetime import datetime, timedelta, date

NOW = datetime.now()


def get_now() -> datetime:
    return NOW


def to_datetime(d: date):
    return datetime.combine(d, datetime.min.time())


def ago(delta: timedelta) -> datetime:
    return get_now() - delta


def get_maximum_lookback() -> timedelta:
    return timedelta(days=365)


def get_beginning_of_time() -> datetime:
    return get_now() - get_maximum_lookback()

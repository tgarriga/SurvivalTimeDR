"""Portable run metadata; no scheduler or parent-repository dependency."""


def peak_rss_kib():
    import sys

    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024 if sys.platform == "darwin" else value

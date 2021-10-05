from prepare import prep
from acquire import acquire


def wrangle():
    df = prep(acquire())
    return df
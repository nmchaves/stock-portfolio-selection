"""
    Class to represent stock market data using price relatives.
    The current literature tends to used relative price changes, so we
    adopt this convention as well.
"""

import util


class MarketData:
    def __init__(self, vol, op, lo, hi, cl, stocks):
        # Store prices in both raw and relative formats
        self.raw = {
            'vol': vol,
            'op': op,
            'lo': lo,
            'hi': hi,
            'cl': cl,
        }
        self.relative = {
            'vol': util.get_price_relatives(vol),
            'op': util.get_price_relatives(op),
            'lo': util.get_price_relatives(lo),
            'hi': util.get_price_relatives(hi),
            'cl': util.get_price_relatives(cl)
        }
        self.stock_names = stocks

    def get_vol(self, relative=True):
        """
        :param relative: If True, get the relative values. Otherwise
        get the raw values.
        """
        if relative:
            return self.relative['vol']
        else:
            return self.raw['vol']

    def get_op(self, relative=True):
        """
        :param relative: If True, get the relative values. Otherwise
        get the raw values.
        """
        if relative:
            return self.relative['op']
        else:
            return self.raw['op']

    def get_lo(self, relative=True):
        """
        :param relative: If True, get the relative values. Otherwise
        get the raw values.
        """
        if relative:
            return self.relative['lo']
        else:
            return self.raw['lo']

    def get_hi(self, relative=True):
        """
        :param relative: If True, get the relative values. Otherwise
        get the raw values.
        """
        if relative:
            return self.relative['hi']
        else:
            return self.raw['hi']

    def get_cl(self, relative=True):
        """
        :param relative: If True, get the relative values. Otherwise
        get the raw values.
        """
        if relative:
            return self.relative['cl']
        else:
            return self.raw['cl']


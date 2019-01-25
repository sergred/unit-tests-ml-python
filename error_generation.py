#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from copy import deepcopy
import numpy as np

from selection import RandomSelector, PairSelector


class ErrorGenerator:
    def __init__(self):
        self.name = 'error generator'

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def on(self, data, columns=None, row_fraction=.2):
        return self.run(data, columns, row_fraction)

    def run(self, data, columns=None, row_fraction=.2):
        return data


class Anomalies(ErrorGenerator):
    def __init__(self):
        ErrorGenerator.__init__(self)
        self.name = 'numeric anomalies'

    def get_anomaly(self, mean, std):
        factor = np.random.random() - .5
        return mean + np.sign(factor)*(3. + 2 * factor)*std

    def on(self, data, columns=None, row_fraction=.2):
        return self.run(data, columns, row_fraction)

    def run(self, data, columns=None, row_fraction=.2):
        data = deepcopy(data)
        selector = RandomSelector(row_fraction=row_fraction).on(data, columns)
        for col in selector.keys():
            idx = selector[col]
            mean, std = np.mean(data.iloc[:, col]), np.std(data.iloc[:, col])
            data.iloc[idx, col] = (np.vectorize(self.get_anomaly)(mean, std)
                                   .astype(data.dtypes[col]))
        return data


class ExplicitMissingValues(ErrorGenerator):
    def __init__(self):
        ErrorGenerator.__init__(self)
        self.name = 'explicit misvals'

    def apply(self, function, data, cell_ids):
        data = deepcopy(data)
        for col, idx in cell_ids.items():
            data.iloc[idx, col] = np.vectorize(function)(data.iloc[idx, col])
        return data

    def run(self, data, columns=None, row_fraction=.2):
        return self.apply(
            lambda x: np.nan, data, RandomSelector(
                row_fraction=row_fraction).on(data, columns))


class ImplicitMissingValues(ErrorGenerator):
    def __init__(self):
        ErrorGenerator.__init__(self)
        self.name = 'implicit misvals'

    def apply(self, function, data, cell_ids):
        data = deepcopy(data)
        for col, idx in cell_ids.items():
            data.iloc[idx, col] = np.vectorize(function)(data.iloc[idx, col])
        return data

    def run(self, data, columns=None, row_fraction=.2):
        tmp = self.apply(
            lambda x: 9999,
            data, RandomSelector(
                row_fraction=row_fraction).on(data, columns='numeric'))
        return self.apply(
            lambda x: 'undefined',
            tmp, RandomSelector(
                row_fraction=row_fraction).on(data, columns='string'))


class Typos(ErrorGenerator):
    class __Typos:
        def __init__(self):
            self.name = 'typos'
            self.keyApprox = {
                'q': "qwasedzx",
                'w': "wqesadrfcx",
                'e': "ewrsfdqazxcvgt",
                'r': "retdgfwsxcvgt",
                't': "tryfhgedcvbnju",
                'y': "ytugjhrfvbnji",
                'u': "uyihkjtgbnmlo",
                'i': "iuojlkyhnmlp",
                'o': "oipklujm",
                'p': "plo['ik",
                'a': "aqszwxwdce",
                's': "swxadrfv",
                'd': "decsfaqgbv",
                'f': "fdgrvwsxyhn",
                'g': "gtbfhedcyjn",
                'h': "hyngjfrvkim",
                'j': "jhknugtblom",
                'k': "kjlinyhn",
                'l': "lokmpujn",
                'z': "zaxsvde",
                'x': "xzcsdbvfrewq",
                'c': "cxvdfzswergb",
                'v': "vcfbgxdertyn",
                'b': "bvnghcftyun",
                'n': "nbmhjvgtuik",
                'm': "mnkjloik"}

    instance = None

    def __init__(self):
        if not Typos.instance:
            Typos.instance = Typos.__Typos()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def apply(self, function, data, cell_ids):
        data = deepcopy(data)
        for col, idx in cell_ids.items():
            data.iloc[idx, col] = (np.vectorize(function)(data.iloc[idx, col])
                                   .astype(data.dtypes[col]))
        return data

    def run(self, data, columns=None, row_fraction=.2):
        return self.apply(
            self.butterfinger,
            data, RandomSelector(
                row_fraction=row_fraction).on(data, columns='string'))

    def butterfinger(self, text, prob=.05):
        def foo(letter):
            if letter.lower() in self.keyApprox.keys():
                cond = np.random.random() <= prob
                tmp = np.random.choice(
                    list(self.keyApprox[letter.lower()])) if cond else letter
                return tmp.upper() if letter.isupper() else tmp
            return letter
        return np.array("".join(map(foo, text)))


class SwapFields(ErrorGenerator):
    def __init__(self):
        ErrorGenerator.__init__(self)
        self.name = 'swap fields'

    def apply(self, function, data, cell_ids):
        df = deepcopy(data)
        ((lc, lr), (rc, rr)) = cell_ids.items()
        # TODO: swap cols
        (df.iloc[lr, lc], df.iloc[rr, rc]) = (df.iloc[rr, rc].values,
                                              df.iloc[lr, lc].values)
        return df

    def run(self, data, columns=None, row_fraction=.2):
        # print(columns)
        return self.apply(None, data, PairSelector(
            row_fraction=row_fraction).on(data, columns))


def main():
    """
    """
    error_gen = Typos()
    print(error_gen.butterfinger("Hello World"))
    print(np.vectorize(error_gen.butterfinger)(["Hello",  "World"]))


if __name__ == "__main__":
    main()

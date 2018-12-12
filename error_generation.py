#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from selection import RandomSelector
from copy import deepcopy
import numpy as np


class ErrorGenerator:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def on(self, data, columns=None):
        return self.run(data, columns)

    def run(self, data, columns=None):
        return data


class Anomalies(ErrorGenerator):
    def __init__(self):
        ErrorGenerator.__init__(self)
        pass

    def get_anomaly(self, mean, std):
        factor = np.random.random() - .5
        return mean + np.sign(factor)*(3. + 2 * factor)*std

    def on(self, data, columns=None):
        return self.run(data, columns)

    def run(self, data, columns=None):
        data = deepcopy(data)
        selector = RandomSelector().on(data, columns)
        for col in selector.keys():
            idx = selector[col]
            mean, std = np.mean(data.iloc[:, col]), np.std(data.iloc[:, col])
            data.iloc[idx, col] = (np.vectorize(self.get_anomaly)(mean, std)
                                   .astype(data.dtypes[col]))
        return data


class ExplicitMissingValues(ErrorGenerator):
    def __init__(self):
        ErrorGenerator.__init__(self)
        pass

    def apply(self, function, data, cell_ids):
        data = deepcopy(data)
        for col, idx in cell_ids.items():
            data.iloc[idx, col] = np.vectorize(function)(data.iloc[idx, col])
        return data

    def run(self, data, columns=None):
        return self.apply(
            lambda x: np.nan, data, RandomSelector().on(data, columns))


class ImplicitMissingValues(ErrorGenerator):
    def __init__(self):
        ErrorGenerator.__init__(self)
        pass

    def apply(self, function, data, cell_ids):
        data = deepcopy(data)
        for col, idx in cell_ids.items():
            data.iloc[idx, col] = np.vectorize(function)(data.iloc[idx, col])
        return data

    def run(self, data, columns=None):
        tmp = self.apply(
            lambda x: 9999,
            data, RandomSelector().on(data, columns='numeric'))
        return self.apply(
            lambda x: 'undefined',
            tmp, RandomSelector().on(data, columns='string'))


class Typos(ErrorGenerator):
    class __Typos:
        def __init__(self):
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

    def run(self, data, columns=None):
        return self.apply(
            self.butterfinger,
            data, RandomSelector().on(data, columns='string'))

    def butterfinger(self, text, prob=.05):
        def foo(letter):
            if letter.lower() in self.keyApprox.keys():
                cond = np.random.random() <= prob
                tmp = np.random.choice(
                    list(self.keyApprox[letter.lower()])) if cond else letter
                return tmp.upper() if letter.isupper() else tmp
            return letter
        return np.array("".join(map(foo, text)))


def main():
    """
    """
    error_gen = Typos()
    print(error_gen.butterfinger("Hello World"))
    print(np.vectorize(error_gen.butterfinger)(["Hello",  "World"]))


if __name__ == "__main__":
    main()

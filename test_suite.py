#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

import numpy as np

from profilers import ErrorType, Severity, DataFrameProfiler
from analyzers import DataScale


class Check:
    def __init__(self, assumption, error_type, severity, message):
        self.assumption = assumption
        self.error_type = error_type
        self.message = message.strip()
        self.severity = severity

    def __repr__(self):
        return """Check:{:s} - IF {:s}""".format(self.error_type.name,
                                                 self.message)

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((self.error_type.value,
                     self.message, self.severity.value))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class Warning:
    def __init__(self, error_type, severity, message=""):
        self.error_type = error_type
        self.message = message.strip()
        self.severity = severity

    def __repr__(self):
        return "{:10}: {:s}".format(self.severity.name, self.message)

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((self.error_type.value, self.message,
                     self.severity.value))

    def __eq__(self, other):
        # return self.__hash__() == other.__hash__()
        a = (self.error_type.value == other.error_type.value)
        b = (self.message == other.message)
        c = (self.severity.value == other.severity.value)
        return a and b and c


class Test:
    def __init__(self, severity=Severity.CRITICAL):
        self.checks = []
        self.severity = severity

    def __iter__(self):
        return iter(self.checks)

    def __repr__(self):
        checks = ", ".join(map(str, self.checks))
        return """Test:{:10} ({:s})""".format(self.severity.name, checks)

    def __str__(self):
        return self.__repr__()

    def _add(self, assumption, error_type, message, severity=None):
        severity = severity if severity is not None else self.severity
        self.checks.append(Check(assumption, error_type, severity, message))
        return self

    def has_size(self, size):
        return self._add(lambda x: x.shape[0] == size,
                         ErrorType.INTEGRITY, """
                         DataFrame does not have %d rows""" % size)

    def is_complete(self, column):
        return self._add(lambda x: x[column].notna().all(),
                         ErrorType.MISSING_VALUE, """
                         Column %s is not complete""" % column)

    def is_unique(self, column):
        return self._add(lambda x: np.unique(x[column]).shape[0] == x.shape[0],
                         ErrorType.DUPLICATE, """
                         Values in column %s are not unique""" % column)

    def is_in_range(self, column, range):
        return self._add(lambda x: x[column].isin(range).all(),
                         ErrorType.NOT_IN_RANGE, """
                         Values in column %s are not in range %s."""
                         % (column, str(range)))

    def __hash__(self):
        return hash((self.severity.value, tuple(self.checks)))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class TestSuite:
    def __init__(self):
        self.tests = []

    def on(self, data):
        return self.run(data)

    def add(self, test):
        self.tests.append(test)
        return self

    def run(self, data):
        assert data is not None, "Call TestSuite on(data) method first."
        warnings = []
        for test in self.tests:
            for check in test:
                if check.assumption(data):
                    continue
                warnings.append(Warning(check.error_type,
                                        Severity.CRITICAL, check.message))
        return warnings


class AutomatedTestSuite:
    class __PlaceHolder:
        def __init__(self):
            self.mes_unavailable_data_profile = """
            Cannot analyze the dataset. Data profile is not available.
            """
            self.mes_unavailable_pipeline_profile = """
            Cannot analyze the pipeline. Pipeline profile is not available.
            """
            self.col_wise = dict({
                'missing_values': lambda x: self.missing_values(x)
            })
            self.nominal_col = dict({
                'values_in_range': lambda x: Test(
                    Severity.CRITICAL).is_in_range(x.column_name, x.range)
            })
            self.ordinal_col = dict({

            })
            self.interval_col = dict({

            })
            self.ratio_col = dict({

            })
            self.set_wise = dict({

            })
            self.pipe = dict({

            })
            self.scale_wise = dict({
                DataScale.UNDEFINED: dict(),
                DataScale.NOMINAL: self.nominal_col,
                DataScale.ORDINAL: self.ordinal_col,
                DataScale.INTERVAL: self.interval_col,
                DataScale.RATIO: self.ratio_col
            })

        def missing_values(self, col_profile):
            # An error is critical if there weren't any missing values
            # discovered before. If missing values originally present - this
            # issue is considered to be known thus not critical.
            # condition = col_profile.num_missing == 0
            # severity = Severity.CRITICAL if condition else Severity.INFO
            return Test(Severity.CRITICAL).is_complete(col_profile.column_name)

    instance = None

    def __init__(self):
        if not AutomatedTestSuite.instance:
            AutomatedTestSuite.instance = AutomatedTestSuite.__PlaceHolder()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def run(self, data, pipeline_profile):
        data_profile = DataFrameProfiler().on(data)
        tests, warnings = [], []
        if data_profile is not None:
            for column_profile in data_profile:
                # print(column_profile)
                for k, v in self.col_wise.items():
                    tests.append(v(column_profile))
                for k, v in self.scale_wise[column_profile.scale].items():
                    tests.append(v(column_profile))
            for k, v in self.set_wise.items():
                tests.append(v(data_profile))
        else:
            warnings.append(Warning(ErrorType.UNAVAILABLE, Severity.CRITICAL,
                                    self.mes_unavailable_data_profile))
        if pipeline_profile is not None:
            for step in pipeline_profile:
                # print(step)
                for k, v in self.pipe.items():
                    tests.append(v(pipeline_profile))
        else:
            warnings.append(Warning(ErrorType.UNAVAILABLE, Severity.CRITICAL,
                                    self.mes_unavailable_pipeline_profile))
        for test in tests:
            for check in test.checks:
                if check.assumption(data):
                    continue
                warnings.append(
                    Warning(check.error_type, test.severity, check.message))
        return tests, warnings


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()

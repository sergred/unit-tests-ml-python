#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

import numpy as np

from profilers import Warning, ErrorType, Severity
from analyzers import DataScale


class Check:
    def __init__(self, assumption, error_type, severity, message):
        self.assumption = assumption
        self.error_type = error_type
        self.message = message
        self.severity = severity


class Test:
    def __init__(self, severity=Severity.CRITICAL):
        self.checks = []
        self.severity = severity

    def __iter__(self):
        return iter(self.checks)

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
                         ErrorType.INCOMPLETE, """
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


class TestSuite:
    def __init__(self):
        self.data = None
        self.tests = []

    def on(self, data):
        self.data = data
        return self

    def add(self, test):
        self.tests.append(test)
        return self

    def run(self):
        assert self.data is not None, "Call TestSuite on(data) method first."
        self.warnings = []
        for test in self.tests:
            for check in test:
                if check.assumption(self.data):
                    continue
                self.warnings.append(Warning(check.error_type,
                                             Severity.CRITICAL, check.message))
        return self.warnings


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
                'values_in_range': lambda x: self.values_in_range(x)
            })
            self.ordinal_col = dict({
                'values_in_range': lambda x: self.values_in_range(x)
            })
            self.interval_col = dict({
                'values_in_range': lambda x: self.values_in_range(x)
            })
            self.ratio_col = dict({
                'values_in_range': lambda x: self.values_in_range(x)
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
            condition = col_profile.num_missing == 0
            severity = Severity.CRITICAL if condition else Severity.INFO
            return (Test(severity).is_complete(col_profile.column_name),
                    Warning(ErrorType.MISSING_VALUE, severity, """
                            Column %s contains missing values.
                            """ % col_profile.column_name))

        def values_in_range(self, col_profile):
            return (Test(Severity.CRITICAL)
                    .is_in_range(col_profile.column_name, col_profile.range),
                    Warning(ErrorType.NOT_IN_RANGE, Severity.CRITICAL, """
                            Values in column %s are not in range %s.
                            """ % (col_profile.column_name,
                                   col_profile.range)))

    instance = None

    def __init__(self):
        if not AutomatedTestSuite.instance:
            AutomatedTestSuite.instance = AutomatedTestSuite.__PlaceHolder()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def run(self, data_profile, pipeline_profile):
        tests, warnings = [], []
        if data_profile is not None:
            for column_profile in data_profile:
                print(column_profile)
                for k, v in self.col_wise.items():
                    print(k)
                    test, warning = v(column_profile)
                    tests.append(test)
                    warnings.append(warning)
                for k, v in self.scale_wise[column_profile.scale].items():
                    print(k)
                    test, warning = v(column_profile)
                    tests.append(test)
                    warnings.append(warning)
            for k, v in self.set_wise.items():
                print(k)
                test, warning = v(data_profile)
                tests.append(test)
                warnings.append(warning)
        else:
            warnings.append(Warning(ErrorType.UNAVAILABLE, Severity.CRITICAL,
                                    self.mes_unavailable_data_profile))
        if pipeline_profile is not None:
            for step in pipeline_profile:
                print(step)
                for k, v in self.pipe.items():
                    print(k)
                    v()
        else:
            warnings.append(Warning(ErrorType.UNAVAILABLE, Severity.CRITICAL,
                                    self.mes_unavailable_pipeline_profile))
        return tests, warnings


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()

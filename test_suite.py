#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

import numpy as np

from profilers import ErrorType, Severity
from analyzers import DataScale
from messages import Message


class Check:
    def __init__(self, assumption, error_type, severity, message):
        self.assumption = assumption
        self.error_type = error_type
        self.message = message.strip()
        self.severity = severity

    def __repr__(self):
        return Message().check.format(self.error_type.name, self.message)

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
        return self.__hash__() == other.__hash__()
        # return ((self.error_type.value == other.error_type.value)
        #         and (self.message == other.message)
        #         and (self.severity.value == other.severity.value))


class Test:
    def __init__(self, severity=Severity.CRITICAL):
        self.checks = []
        self.severity = severity

    def __iter__(self):
        return iter(self.checks)

    def __repr__(self):
        checks = ", ".join(map(str, self.checks))
        return Message().test.format(self.severity.name, checks)

    def __str__(self):
        return self.__repr__()

    def _add(self, assumption, error_type, message, severity=None):
        severity = severity if severity is not None else self.severity
        self.checks.append(Check(assumption, error_type, severity, message))
        return self

    def has_size(self, profile):
        return self._add(lambda x: x.shape[0] == profile.size,
                         ErrorType.INTEGRITY,
                         Message().wrong_size % profile.size)

    def is_complete(self, profile):
        return self._add(
            lambda x: x[profile.column_name].notna().all(),
            ErrorType.MISSING_VALUE,
            Message().not_complete % profile.column_name)

    def is_unique(self, profile):
        return self._add(
            lambda x: np.unique(x[profile.column_name]).shape[0] == x.shape[0],
            ErrorType.DUPLICATE, Message().not_unique % profile.column_name)

    def is_in_range(self, profile):
        return self._add(
            lambda x: (x[profile.column_name].isin(profile.range).all()
                       if profile.range is not None else True),
            ErrorType.NOT_IN_RANGE,
            Message().not_in_range % (profile.column_name, str(profile.range)))

    def is_non_negative(self, profile):
        pass

    def contains_url(self, profile):
        pass

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
        assert data is not None, Message().no_testsuite
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
            self.col_wise = dict({
                'missing_values': lambda x: (Test(Severity.CRITICAL)
                                             .is_complete(x))
            })
            self.nominal_col = dict({
                'values_in_range': lambda x: (Test(Severity.CRITICAL)
                                              .is_in_range(x))
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
            self.tests = None
            self.warnings = None

    instance = None

    def __init__(self):
        if not AutomatedTestSuite.instance:
            AutomatedTestSuite.instance = AutomatedTestSuite.__PlaceHolder()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def on(self, data):
        return self.run(data)

    def with_profiles(self, data_profile, pipeline_profile):
        self.tests, self.warnings = [], []
        if data_profile is not None:
            for column_profile in data_profile.profiles:
                # print(column_profile)
                for test in self.col_wise.values():
                    self.tests.append(test(column_profile))
                for test in self.scale_wise[column_profile.scale].values():
                    self.tests.append(test(column_profile))
            for test in self.set_wise.values():
                self.tests.append(test(data_profile))
        else:
            self.warnings.append(
                Warning(ErrorType.UNAVAILABLE, Severity.CRITICAL,
                        Message().data_profile_not_available))
        if pipeline_profile is not None:
            for step in pipeline_profile:
                # print(step)
                for test in self.pipe.values():
                    self.tests.append(test(pipeline_profile))
        else:
            self.warnings.append(
                Warning(ErrorType.UNAVAILABLE, Severity.CRITICAL,
                        Message().pipeline_profile_not_available))
        return self

    def get_tests(self):
        assert self.tests is not None, Message().no_profile
        return self.tests

    def run(self, data):
        assert self.tests is not None, Message().no_profile
        assert self.warnings is not None, Message().no_profile
        for test in self.tests:
            for check in test.checks:
                if check.assumption(data):
                    continue
                self.warnings.append(
                    Warning(check.error_type, test.severity, check.message))
        return self.tests, self.warnings


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()

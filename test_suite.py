#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

from profilers import Warning, ErrorType

class Check:
    def __init__(self, assumption, error_type, message):
        self.assumption = assumption
        self.error_type = error_type
        self.message = message


class Test:
    def __init__(self):
        self.checks = []

    def __iter__(self):
        return iter(self.checks)

    def _add(self, assumption, error_type, message):
        self.checks.append(Check(assumption, error_type, message))
        return self

    def has_size(self, size):
        return self._add(lambda x: x.shape[0] == size,
                         ErrorType.INTEGRITY,
                         "DataFrame does not have %d rows" % size)

    def is_complete(self, column):
        return self._add(lambda x: x[column].notna().all(),
                         ErrorType.INCOMPLETE,
                         "Column %s is not complete" % column)

    def is_unique(self, column):
        return self._add(lambda x: np.unique(x[column]).shape[0] == x.shape[0],
                         ErrorType.DUPLICATE,
                         "Values in column %s not not unique" % column)


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
                if check.assumption(self.data): continue
                self.warnings.append(Warning(check.error_type, check.message))
        return self.warnings


class AutomatedTestSuite:
    def __init__(self, data_profile, pipeline_profile):
        self.tests = []
        self.warnings = []
        self.data_profile = data_profile
        self.pipeline_profile = pipeline_profile

    def run(self):
        if self.data_profile is not None:
            for column_profile in self.data_profile:
                # print(column_profile)
                pass
        if self.pipeline_profile is not None:
            for step in self.pipeline_profile:
                print(step)
        return self.tests, self.warnings


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()

#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

"""
val verificationResult = VerificationSuite()
    .onData(data)
    .addCheck(
        Check(CheckLevel.Error, "unit testing my data")
        .hasSize(_ == 5) // we expect 5 rows
        .isComplete("id") // should never be NULL
        .isUnique("id") // should not contain duplicates
        .isComplete("name") // should never be NULL
        // should only contain the values "high" and "low"
        .isContainedIn("priority", Array("high", "low"))
        .isNonNegative("numViews") // should not contain negative values
        // at least half of the descriptions should contain a url
        .containsURL("description", _ >= 0.5)
        // half of the items should have less than 10 views
        .hasApproxQuantile("numViews", 0.5, _ <= 10))
    .run()
"""

from enum import Enum

class CheckLevel(Enum):
    Error = 1


class Check:
    def __init__(self, check_level, description):
        self.check_level = check_level
        self.description = description

    def has_size(self, size):
        pass

    def is_complete(self, column):
        pass

    def is_unique(self, column):
        pass

    def is_contained_in(self, column, set):
        pass

    def is_non_negative(self, column):
        pass

    def contains_url(self, column, ratio=1.):
        pass

    def has_approx_quantile(self, column, quantile):
        pass


class VerificationResult:
    def __init__(self):
        pass


class VerificationSuite:
    def __init__(self):
        pass

    def on_data(self, data):
        pass

    def add_check(self, check):
        pass

    def run(self):
        result = VerificationResult()
        return result


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()

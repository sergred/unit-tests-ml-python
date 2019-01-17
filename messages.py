#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""


class Message:
    class __PlaceHolder:
        def __init__(self):
            self.check = "Check:{:s} - IF {:s}"
            self.test = "Test:{:10} ({:s})"
            self.precision = "Precision: %0.2f (+/- %0.2f)"
            self.accuracy = "Accuracy: %0.2f (+/- %0.2f)"
            self.recall = "Recall: %0.2f (+/- %0.2f)"
            self.wrong_size = "DataFrame does not have %d rows"
            self.wrong_value = "Value %s is not correct"
            self.not_complete = "Column %s is not complete"
            self.not_unique = "Values in column %s are not unique"
            self.not_in_range = "Values in column %s are not in range %s"
            self.not_a_pipeline = "sklearn.pipeline.Pipeline required"
            self.not_enough_history = """
            Call .iteration method to get more history""".strip()
            self.no_testsuite = "Call TestSuite on(data) method first."
            self.no_profile = "Call .with_profiles method first."
            self.no_column = "Column %s does not exist"
            self.data_profile_not_available = """
            Cannot analyze the dataset. Data profile is not available.
            """
            self.pipeline_profile_not_available = """
            Cannot analyze the pipeline. Pipeline profile is not available.
            """

    instance = None

    def __init__(self):
        if not Message.instance:
            Message.instance = Message.__PlaceHolder()

    def __getattr__(self, name):
        return getattr(self.instance, name)


def main():
    """
    """
    pass


if __name__ == "__main__":
    main()

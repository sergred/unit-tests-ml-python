#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""""""

import unittest

from pipelines import CreditGPipeline


class EvaluationSuite(unittest.TestSuite):
    def __init__(self):
        pass

    def run(self):
        # with self.assertRaises(SomeException) as cm:
        #     do_something()

        # the_exception = cm.exception
        # self.assertEqual(the_exception.error_code, 3)
        pass


class CreditGTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = CreditGPipeline()

    def tearDown(self):
        pass

    def test_(self):
        pass

# class Test(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass


if __name__ == '__main__':
    unittest.main()

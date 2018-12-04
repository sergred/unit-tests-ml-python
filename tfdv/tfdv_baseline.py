#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

""""""

from sklearn.model_selection import train_test_split
import tensorflow_data_validation as tfdv
from google.protobuf import text_format
import apache_beam as beam
import tensorflow as tf
import pandas as pd
import numpy as np
import os

import sys
sys.path.append(os.path.dirname(os.getcwd()))

from analyzers import DataType
from error_generation import ExplicitMissingValues

sys.path.append(
    os.path.join(os.path.dirname(os.getcwd()), 'third_party/data-linter'))

from feature_statistics_pb2 import DatasetFeatureStatisticsList
import lint_explorer
import explanations
import data_linter
import example_pb2
import linters

sys.path.append(
    os.path.join(os.path.dirname(os.getcwd()),
                 'third_party/facets/facets_overview/python'))

from feature_statistics_generator import ProtoFromTfRecordFiles

np.random.seed = 1

# Some linters are currently disabled due to a bug.
DEFAULT_STATS_LINTERS = [  # These linters require dataset statistics.
    linters.CircularDomainDetector,
    linters.DateTimeAsStringDetector,
    linters.DuplicateExampleDetector,
#    linters.EnumDetector,
#    linters.IntAsFloatDetector,
    linters.NonNormalNumericFeatureDetector,
    linters.NumberAsStringDetector,
    linters.TailedDistributionDetector,
    linters.TokenizableStringDetector,
    linters.UncommonListLengthDetector,
#    linters.UncommonSignDetector,
    linters.ZipCodeAsNumberDetector,
]

DEFAULT_LINTERS = [
    linters.EmptyExampleDetector,
]


def _ensure_directory_exists(path):
    directory_path = os.path.dirname(path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def _make_dataset_reader(dataset_path, example_coder):
    """Returns the appropriate reader for the dataset.
    Args:
    dataset_path: The path (or glob) to the dataset files.
    example_coder: A `ProtoCoder` for `tf.Example`s
    Returns:
    A `LabeledPTransform` that yields a `PCollection` of the
    `tf.Example`s in the dataset.
    """
    reader = beam.io.tfrecordio.ReadFromTFRecord(dataset_path, coder=example_coder)
    return 'ReadExamples' >> reader


def _read_feature_stats(stats_path):
    with open(stats_path) as fin:
        summaries = DatasetFeatureStatisticsList()
        summaries.ParseFromString(fin.read())
    return summaries.datasets[0]


class TFRecordHelper:
    class __TFRecordHelper:
        def __init__(self):
            self.foo = dict({
                DataType.STRING: lambda x, y: x.bytes_list.value.extend([y]),
                DataType.INTEGER: lambda x, y: x.int64_list.value.extend([y]),
                DataType.FLOAT: lambda x, y: x.float_list.value.extend([y]),
                DataType.OBJECT: lambda x, y: x.bytes_list.value.extend([y])
            })
            self.data_type = dict({
                'int': DataType.INTEGER,
                'int32': DataType.INTEGER,
                'int64': DataType.INTEGER,
                'float': DataType.FLOAT,
                'float32': DataType.FLOAT,
                'float64': DataType.FLOAT,
                'byte': DataType.OBJECT,
                # 'string': DataType.STRING,
                'object': DataType.OBJECT
            })

        def run(self, example, feature_name, dtype, val):
            if not isinstance(dtype, DataType):
                dtype = self.data_type[str(dtype)]
            return self.foo[dtype](example.features.feature[feature_name], val)

    instance = None

    def __init__(self):
        if not TFRecordHelper.instance:
            TFRecordHelper.instance = TFRecordHelper.__TFRecordHelper()

    def __getattr__(self, name):
        return getattr(self.instance, name)


def convert_csv_to_tfrecord(data_path, file_name, dtypes=None):
    filename = os.path.join(data_path, file_name.split('.')[0] + '.tfrecords')
    data = pd.read_csv(os.path.join(data_path, file_name))
    helper = TFRecordHelper()
    columns = data.columns
    if dtypes is None:
        dtypes = data.dtypes
    with tf.python_io.TFRecordWriter(filename) as writer:
        for i in range(data.shape[0]):
            example = tf.train.Example()
            for j in range(data.shape[1]):
                helper.run(example, columns[j], dtypes[j], data.iloc[i, j])
            writer.write(example.SerializeToString())


def train_test_split_csv(data_path, file_name):
    data = pd.read_csv(os.path.join(data_path, file_name))
    train, test = train_test_split(data, test_size=0.33, random_state=1)
    train.to_csv(os.path.join(data_path, 'train.csv'))
    test.to_csv(os.path.join(data_path, 'test.csv'))


def data_validation(data_path):
    train = tfdv.generate_statistics_from_csv(
        os.path.join(data_path, 'train.csv'), delimiter=',')
    test = tfdv.generate_statistics_from_csv(
        os.path.join(data_path, 'train.csv'), delimiter=',')
    schema = tfdv.infer_schema(train)
    # print(schema)
    # tfdv.display_schema(schema)
    anomalies = tfdv.validate_statistics(statistics=test, schema=schema)
    # print(anomalies)
    # tfdv.display_anomalies(anomalies)
    print(text_format.MessageToString(anomalies))


def main():
    """
    """
    data_path = os.path.join('../resources/data/', 'wine-quality')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_name = 'wine-quality-red.csv'
    convert_csv_to_tfrecord(data_path, file_name)
    # train_test_split_csv(data_path, file_name)
    err_gen = ExplicitMissingValues()
    new_test = err_gen.on(pd.read_csv(os.path.join(data_path, 'test_old.csv')))
    print(pd.isnull(new_test).any())
    new_test.to_csv(os.path.join(data_path, 'test.csv'))
    data_validation(data_path)

    DATASET_PATH = os.path.join(data_path, 'wine-quality-red.tfrecords')
    OUTPUT_PATH = os.path.join(data_path, 'stats.bin')
    DATASET_NAME = 'wine-quality'

    result = ProtoFromTfRecordFiles(
        [{"name": DATASET_NAME, "path": DATASET_PATH}], max_entries=1000000)
    with open(OUTPUT_PATH, "w") as fout:
        fout.write(result.SerializeToString())

    stats_path = os.path.join(data_path, 'stats.bin')
    results_path = os.path.join(data_path, 'results.tfrecords')
    if not os.path.exists(stats_path):
        raise ValueError('Error: stats path %s does not exist' % stats_path)

    stats = _read_feature_stats(stats_path)

    run_linters = [stats_linter(stats) for stats_linter in DEFAULT_STATS_LINTERS]
    run_linters.extend([linter() for linter in DEFAULT_LINTERS])
    datalinter = data_linter.DataLinter(run_linters, results_path)

    _ensure_directory_exists(results_path)
    with beam.Pipeline() as p:
        _ = (
            p
            | _make_dataset_reader(DATASET_PATH,
                                   beam.coders.ProtoCoder(example_pb2.Example))
            | 'LintData' >> datalinter)

    lint_results = lint_explorer.load_lint_results(results_path)
    suppressed_warnings = lint_explorer.suppress_warnings(lint_results)
    disp_results = lint_explorer.format_results(lint_results,
                                                suppressed_warnings)
    print(disp_results)


if __name__ == "__main__":
    main()

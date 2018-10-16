#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

"""Settings getter functions."""

import json
import os

def settings():
    """Settings getter."""
    with open(os.path.join(os.getcwd(), 'settings.json')) as data_file:
        settings = json.load(data_file)
    return settings


def get_param(model_name, param_name):
    """
    Model parameter getter.

    Keyword arguments:
    model_name -- model name
    param_name -- parameter name
    """
    if model_name in settings()["models"]:
        if param_name in settings()[model_name]:
            """Returns values or absolute paths"""
            if param_name.split('_')[-1] == 'path':
                return os.path.join(os.getcwd(), settings()[model_name][param_name])
            else:
                return settings()[model_name][param_name]
        else:
            raise ValueError("Unknown parameter.")
    else:
        raise ValueError("Unknown model name.")


def get_auth_string():
    return settings()["auth_string"]


def get_resource_path():
    return settings()["resource_path"]

def main():
    """Getters test function."""
    pass


if __name__ == "__main__":
    main()

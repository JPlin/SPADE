import importlib


def create_options(option_name):
    # Given the option name
    # the file "options/option_name_options.py"
    # will be imported
    option_filename = "options." + option_name + "_options"
    optionlib = importlib.import_module(option_filename)
    xoption = optionlib.Xoptions()
    xoption.opt['name'] = option_name
    return xoption
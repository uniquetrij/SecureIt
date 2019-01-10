from os.path import dirname, realpath


def get():
    return dirname(realpath(__file__))
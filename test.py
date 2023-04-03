#!/usr/bin/env python
from __future__ import print_function
from parser import get_parser
from Processor import Processor


if __name__ == '__main__':
    arg = get_parser()
    processor = Processor(arg)
    processor.start()
    print("done")
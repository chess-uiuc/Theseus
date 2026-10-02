#!/usr/bin/env python3
"""Apply runner placement defaults without duplicating test-supplied options."""
import json
import os
import sys


PLACEMENT_OPTIONS = {'--host', '--map-by', '--bind-to'}


def merge_arguments(defaults, supplied):
    explicit = {arg.split('=', 1)[0] for arg in supplied} & PLACEMENT_OPTIONS
    result = []
    index = 0
    while index < len(defaults):
        arg = defaults[index]
        option = arg.split('=', 1)[0]
        width = 2 if option in PLACEMENT_OPTIONS and '=' not in arg else 1
        if option not in explicit:
            result.extend(defaults[index:index + width])
        index += width
    return result + supplied


if __name__ == '__main__':
    launcher, defaults = sys.argv[1:3]
    os.execv(launcher, [launcher, *merge_arguments(json.loads(defaults), sys.argv[3:])])

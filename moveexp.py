# -*- coding: utf-8 -*-
"""

Created on April 21, 2019

@author:  neerbek
"""
import os
import argparse
import sh

# os.chdir("DeepTorch")

parser = argparse.ArgumentParser()
parser.add_argument("--allow_update", help="Allow moveexp to update existing log files in destination", action='store_true', default=False)
parser.add_argument("--allow_active_screens", help="Allow moveexp to run even if active screens/exp are runinng", action='store_true', default=False)
parser.add_argument("exp", help="Exp to move",
                    type=str)
# s = "tmp"
# args = parser.parse_args(s.split())
args = parser.parse_args()

outPath = os.path.join(os.getenv("HOME"), "jan/ProjectsData/phd/DLP/Monsanto/data/logs")

if not os.path.exists(outPath):
    raise Exception("ProjectsData path not found")

outPath = os.path.join(outPath, args.exp)

if not os.path.exists(outPath):
    os.mkdir(outPath)
else:
    if args.allow_update:
        print("Exp destination exists, updating files")
    else:
        raise Exception("Exp destination exists: {}. Aborting".format(outPath))

hasActiveScreens = True
try:
    sh.screen("-list")
except sh.ErrorReturnCode_1:
    hasActiveScreens = False

# res = "{}".format(screen("-list"))
# res = res.split("\n")
if hasActiveScreens and not args.allow_active_screens:
    raise Exception("active screens exists. Aborting")

mv = sh.mv.bake(_cwd=".")
print(mv(sh.glob('save_{}*'.format(args.exp)), outPath))
print(mv(sh.glob('{}*'.format(args.exp)), outPath))
print("done")

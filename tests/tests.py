#!/usr/bin/python3

#
# Execute this file to run all unit tests of the projet
#

import unittest
import os, sys
from pathlib import Path

def main(args):
    python_file_path = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(python_file_path)
    tests_path = python_file_path
    root_only = False
    
    if len(args):
        if args[0] == "integration":
            print("Run Integration tests")
            tests_path = os.path.join(python_file_path, "integration")
        elif args[0] == "unit":
            print("Run unit tests")
            tests_path = os.path.join(python_file_path, "unit")
        elif args[0] == "steps":
            print("Run step tests")
            tests_path = os.path.join(python_file_path, "steps")
        elif args[0] == "statistics":
            print("Run statistics tests")
            tests_path = os.path.join(python_file_path, "statistics")
        elif args[0] == "root":
            print("Run root tests")
            root_only = True
        else:
            print("Invalid arguments")
            exit(os.EX_DATAERR)   
    else:
        print("Run all tests")
        
    run_test(tests_path, repo_root, root_only=root_only)
            
            

def run_test(tests_path, top_level_dir, root_only=False):
    loader = unittest.TestLoader()
    if root_only:
        package = os.path.relpath(tests_path, top_level_dir).replace(os.sep, ".")
        modules = [
            f"{package}.{path.stem}"
            for path in sorted(Path(tests_path).glob("test*.py"))
            if path.is_file()
        ]
        suite = loader.loadTestsFromNames(modules)
    else:
        suite = loader.discover(
            tests_path,
            pattern="test_*.py",
            top_level_dir=top_level_dir,
        )
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if not test_result.wasSuccessful():
        exit(os.EX_SOFTWARE)

if __name__ == "__main__":
    main(sys.argv[1:])

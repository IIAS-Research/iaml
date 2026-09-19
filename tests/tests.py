#!/usr/bin/python3

#
# Execute this file to run all unit tests of the projet
#

import unittest
import os, sys

def main(args):
    python_file_path = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(python_file_path)
    tests_path = python_file_path
    
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
        else:
            print("Invalid arguments")
            exit(os.EX_DATAERR)   
    else:
        print("Run all tests")
        
    run_test(tests_path, repo_root)
            
            

def run_test(tests_path, top_level_dir):
    suite = unittest.TestLoader().discover(
        tests_path,
        pattern="test_*.py",
        top_level_dir=top_level_dir,
    )
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if not test_result.wasSuccessful():
        exit(os.EX_SOFTWARE)

if __name__ == "__main__":
    main(sys.argv[1:])

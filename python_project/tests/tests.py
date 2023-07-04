#!/usr/bin/python3

#
# Execute this file to run all unit tests of the projet
#

import unittest
import os, sys

def main(args):
    python_file_path = os.path.dirname(os.path.abspath(__file__))
    tests_path = python_file_path+"/tests/"
    
    if len(args):
        if args[0] == "integration":
            print("Run Integration tests")
            tests_path = python_file_path+"/tests/integration"
        elif args[0] == "unit":
            print("Run unit tests")
            tests_path = python_file_path+"/tests/unit"
        else:
            print("Invalid arguments")
            exit(os.EX_DATAERR)   
    else:
        print("Run all tests")
        
    run_test(tests_path)
            
            

def run_test(tests_path):
    suite = unittest.TestLoader().discover(tests_path, pattern = "test_*.py")
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if not test_result.wasSuccessful():
        exit(os.EX_SOFTWARE)

if __name__ == "__main__":
    main(sys.argv[1:])

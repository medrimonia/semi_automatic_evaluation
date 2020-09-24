#!/usr/bin/env python3

import argparse
import os

import semi_automatic_evaluator as sae

class DicomDiscoveryAssignment(sae.EvaluationProcess):
    def __init__(self):
        pass

    def getStructure(self):
        return sae.EvalNode("TD Dicom Discovery", children=[
            self.getDefaultRulesTree()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to the directory")
    args = parser.parse_args()

    os.chdir(args.path)

    groups = sae.GroupCollection.discover("./")

    for g in groups:
        dicom_eval = DicomDiscoveryAssignment()
        root = dicom_eval.getStructure()
        root.eval()
        root.syncPoints()
        print(sae.evalToString(root))

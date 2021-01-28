#!/usr/bin/env python3

import argparse
import os
import sys
import math
from os.path import join, dirname
from anytree.importer import JsonImporter

import semi_automatic_evaluator as sae

class ControlAssignment(sae.EvaluationProcess):
    def __init__(self, path, group):
        super().__init__(path, group)

    def _eval(self):
        global ht_module, ctrl_module, traj_module
        # Eval rules respect even if modules are not found
        self.root.eval()

    def getStructure(self):
        return sae.EvalNode("4-Control", children=[
            self.getDefaultRulesTree(),
            self.getCodeTree(),
            sae.EvalNode("Compte rendu", eval_func=sae.manualEval, max_points=12)
            ])

    def getCodeTree(self):
        parts = [
            "PID",
            "OpenLoop Rail",
            "OpenLoop Pendulum",
            "Feed forward"
        ]
        node = sae.EvalNode("Code",
                            children = [sae.EvalNode(part_name, eval_func = sae.manualEval, max_points=2)
                                        for part_name in parts])
        return node

if __name__ == "__main__":
    sae.SAE_LG = "FR"
    #TODO a generic parser should be moved to SAE, then methods would only be
    #  sae.runEvaluation(customClass)
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to the directory")
    args = parser.parse_args()


    groups = sae.GroupCollection.discover(args.path)
    for g in groups:
        json_path = join(args.path, g.getKey() + ".json")
        original_evaluation = None
        if os.path.isfile(json_path):
            print("Importing existing evaluation for the group: " + g.getKey())
            with open(json_path, 'r') as f:
                original_evaluation = JsonImporter().read(f)
        dicom_eval = ControlAssignment(args.path, g)
        root = dicom_eval.run(original_evaluation)
        root.exportToJson(json_path)
        txt_content = sae.evalToString(root)
        txt_path = join(args.path, g.getKey() + ".txt")
        with open(txt_path, "w") as f:
            f.write(txt_content)
        print(txt_content)

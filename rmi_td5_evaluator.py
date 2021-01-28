#!/usr/bin/env python3

import argparse
import os
import sys
import math
from os.path import join, dirname
from anytree.importer import JsonImporter

import semi_automatic_evaluator as sae

class WalkAssignment(sae.EvaluationProcess):
    def __init__(self, path, group):
        super().__init__(path, group)

    def _eval(self):
        global ht_module, ctrl_module, traj_module
        # Eval rules respect even if modules are not found
        self.root.eval()

    def getStructure(self):
        return sae.EvalNode("5-Walk", children=[
            self.getDefaultRulesTree(),
            self.getHexapodTree(),
            self.getQuadrupedRPPTree(),
            self.getReportTree()
            ])

    def getHexapodTree(self):
        nodes = {
            "Paramètres par défaut" : "Inspection+test du fichier hexapod_rpp_walk.json",
            "Période dynamique" : "Fluidité lors du changement dynamique de la fréquence: filtrage=1pts, bon=2pts",
            "Avant-Arrière" : "Stable = 2pt, fonctionnel = 1pt",
            "Latéral" : "Stable = 2pts,  fonctionnel = 1pt",
            "Rotation" : "Stable = 2pts, fonctionnel= 1pt"
        }
        node = sae.EvalNode("Hexapod",
                            children = [sae.EvalNode(name, eval_func = sae.manualEval, max_points=1.5,
                                                     set_up_func= lambda : print(helper))
                                        for name, helper in nodes.items()])
        return node

    def getQuadrupedRPPTree(self):
        nodes = {
            "Choix offsets" : "1 par 1: 2pts, 2 par 2: 1pt",
            "Méchanisme de stabilisation " : "Fluidité lors du changement dynamique de la fréquence: filtrage=1pts, bon=2pts",
            "Avant-Arrière" : "Stable = 2pt, fonctionnel = 1pt",
            "Latéral" : "Stable = 2pts,  fonctionnel = 1pt",
            "Rotation" : "Stable = 2pts, fonctionnel= 1pt"
        }
        node = sae.EvalNode("QuadrupedRPP",
                            children = [sae.EvalNode(name, eval_func = sae.manualEval, max_points=1.5,
                                                     set_up_func= lambda : print(helper))
                                        for name, helper in nodes.items()])
        return node

    def getReportTree(self):
        return sae.EvalNode("Compte-Rendu",
                            children = [
                                sae.EvalNode("Graphiques", eval_func = sae.manualEval, max_points=1,
                                             set_up_func=lambda : print("Utilisation de graphiques appropriées")),
                                sae.EvalNode("Interaction période/vitesse",
                                             eval_func = sae.manualEval, max_points=1,
                                             set_up_func=lambda : print("Mention pas toujours mieux: 0.5pt, illustration: 0.5pt")),
                                sae.EvalNode("QuadrupedYPP", eval_func = sae.manualEval, max_points=1,
                                             set_up_func=lambda : print("Mention problème IK: 0.5 pt, mention singularité 0.5pt")),
                                sae.EvalNode("Appreciation globale", eval_func = sae.manualEval, max_points=3,
                                             set_up_func=lambda : print("Clarté des explications + contenu global."))
                            ])

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
        dicom_eval = WalkAssignment(args.path, g)
        root = dicom_eval.run(original_evaluation)
        root.exportToJson(json_path)
        txt_content = sae.evalToString(root)
        txt_path = join(args.path, g.getKey() + ".txt")
        with open(txt_path, "w") as f:
            f.write(txt_content)
        print(txt_content)

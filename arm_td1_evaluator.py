#!/usr/bin/env python3

import argparse
import os
from os.path import join, dirname

import semi_automatic_evaluator as sae

class DicomDiscoveryAssignment(sae.EvaluationProcess):
    def __init__(self):
        pass

    def _eval(self):
        super()._eval()
        # current_dir = os.getcwd()
        # group_path = os.path.join(os.path.dirname(self.__archive_path), self.
        # os.chdir()

    def getStructure(self):
        return sae.EvalNode("TD Dicom Discovery", children=[
            self.getDefaultRulesTree(),
            self.getFunctionalTree(),
            sae.EvalNode("Qualité du code", 4.0, eval_func= sae.manualEval)
        ])

    def getFunctionalTree(self):
        return sae.EvalNode("Fonctionnalités", children=[
            self.getOpenDicomTree(),
            self.getInformationTree(),
            sae.EvalNode("Affichage d'image", 4.0, eval_func= sae.manualEval),
        ])

    def getOpenDicomTree(self):
        return sae.EvalNode("Ouverture fichier", children=[
            sae.EvalNode("Ouverture classique", 2.0, eval_func=sae.manualEval),
            sae.EvalNode("Filtre sur l'extension", 1.0, eval_func=sae.manualEval),
            sae.EvalNode("Pas de fichier sélectionné", 1.0, eval_func=sae.manualEval)
        ])

    def getInformationTree(self):
        return sae.EvalNode("ShowStats", 4.0,
                            children=[
                                sae.EvalNode("Patient", 1.0, eval_func=sae.manualEval),
                                sae.EvalNode("Transfer Syntax (code)", 0.5, eval_func=sae.manualEval),
                                sae.EvalNode("Transfer Syntax (texte)", 0.5, eval_func=sae.manualEval),
                                sae.EvalNode("Taille de l'image", 1.0, eval_func=sae.manualEval),
                                sae.EvalNode("Valeurs autorisées", 0.5, eval_func=sae.manualEval),
                                sae.EvalNode("Valeurs utilisées", 0.5, eval_func=sae.manualEval),
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
        dicom_eval = DicomDiscoveryAssignment()
        root = dicom_eval.run(args.path, g)
        print(sae.evalToString(root))

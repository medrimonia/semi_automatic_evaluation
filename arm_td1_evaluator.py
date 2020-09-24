#!/usr/bin/env python3

import argparse
import os
from os.path import join, dirname

import semi_automatic_evaluator as sae

class DicomDiscoveryAssignment(sae.EvaluationProcess):
    def __init__(self, path, group):
        super().__init__(path, group)

    def _eval(self):
        super()._eval()
        current_dir = os.getcwd()
        group_path = os.path.join(os.path.dirname(self.__archive_path), self._group.getKey())
        os.chdir(group_path)
        os.mkdir("build")
        os.chdir("build")
        status, out, err = sae.systemCall("qmake --qt=qt5 .. && make")
        os.chdir(current_dir)
        if (status != 0):
            print("Failed to build project with the following error:\n{:}".format(err))
            if not sae.question("Can you build it manually?"):
                sae.setRecursiveMessage("Failed to build")


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
        dicom_eval = DicomDiscoveryAssignment(args.path, g)
        root = dicom_eval.run()
        print(sae.evalToString(root))

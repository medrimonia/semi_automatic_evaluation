#!/usr/bin/env python3

import argparse
import os
from os.path import join, dirname

import semi_automatic_evaluator as sae

class DicomDiscoveryAssignment(sae.EvaluationProcess):
    def __init__(self, path, group):
        super().__init__(path, group)

    def _eval(self):
        # Eval archive etc independently from the rest
        self.root.children[0].eval()
        current_dir = os.getcwd()
        group_path = os.path.join(os.path.dirname(self._archive_path), self._group.getKey())
        os.chdir(group_path)
        if not os.path.isdir("build"):
            os.mkdir("build")
        os.chdir("build")
        print("Building")
        status, out, err = sae.systemCall("qmake --qt=qt5 .. && make")
        os.chdir(current_dir)
        if (status != 0):
            print("Failed to build project with the following error:\n{:}".format(err))
            if not sae.question("Can you build it manually?"):
                for i in range(1,4):
                    sae.setRecursive(self.root.children[i], "Failed to build", 0.0, True)
                return
        binary_path = join(join(group_path, "build"), "dicom_viewer")
        print("Launch binary " + binary_path)
        self.root.eval()


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
            sae.EvalNode("Pas de fichier sélectionné", 1.0, eval_func=sae.manualEval,
                         set_up_func=lambda: print("Try cancel when opening a file")),
            sae.EvalNode("Filtre sur l'extension", 1.0, eval_func=sae.manualEval,
                         set_up_func=lambda: print("Is there a filter to include only dcm files")),
            sae.EvalNode("Ouverture classique", 2.0, eval_func=sae.manualEval,
                         set_up_func=lambda: print("Open test.dcm file"))
        ])

    def getInformationTree(self):
        return sae.EvalNode("ShowStats", 4.0,
                            children=[
                                sae.EvalNode("Patient", 1.0, eval_func=sae.manualEval,
                                             set_up_func=lambda: print("Expected: CQ500-CT-0")),
                                sae.EvalNode("Transfer Syntax (code)", 0.5, eval_func=sae.manualEval,
                                             set_up_func=lambda: print("Expected: 2")),
                                sae.EvalNode("Transfer Syntax (texte)", 0.5, eval_func=sae.manualEval,
                                             set_up_func=lambda: print("Expected: Little Endian Explicit")),
                                sae.EvalNode("Taille de l'image", 1.0, eval_func=sae.manualEval,
                                             set_up_func=lambda: print("512*512*17")),
                                sae.EvalNode("Valeurs autorisées", 0.5, eval_func=sae.manualEval,
                                             set_up_func=lambda: print("[-33792,31743]")),
                                sae.EvalNode("Valeurs utilisées", 0.5, eval_func=sae.manualEval,
                                             set_up_func=lambda: print("Expected: [-3024, 2393]")),
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

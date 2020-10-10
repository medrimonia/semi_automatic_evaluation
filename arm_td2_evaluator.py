#!/usr/bin/env python3

import argparse
import os
from os.path import join, dirname
from anytree.importer import JsonImporter

import semi_automatic_evaluator as sae

class Dicom3DViewerAssignment(sae.EvaluationProcess):
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
                for i in range(1,len(self.root.children)):
                    self.root.children[i].setRecursive("Failed to build", 0.0, True)
                return
        binary_path = join(join(group_path, "build"), "dicom_viewer")
        print("Launch binary " + binary_path)
        self.root.eval()


    def getStructure(self):
        return sae.EvalNode("TD 3D Viewer", children=[
            self.getDefaultRulesTree(),
            self.get2DTree(),
            self.get3DTree()
        ])

    def get2DTree(self):
        return sae.EvalNode("Fonctionnalités 2D", children=[
            sae.EvalNode("Ouverture multi-fichiers", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Open sparse collection, "
                                                     "then open dense collection")),
            self.sanityChecksTree(),
            self.sliceSliderTree(),
            self.informationTree()
        ])

    def sanityChecksTree(self):
        return sae.EvalNode("Vérification fichiers", children=[
            sae.EvalNode("Plusieurs patients", 0.5, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Open various_patients")),
            sae.EvalNode("Duplication d'instance", 0.5, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Open duplicated")),
            sae.EvalNode("Instances manquantes", 0.5, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Open incomplete")),
            sae.EvalNode("Espacement irrégulier", 0.5, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Open inconsistent"))
        ])

    def sliceSliderTree(self):
        return sae.EvalNode("Choix dynamique du layer", children=[
            sae.EvalNode("Fonctionnalité", 2.0, eval_func= sae.manualEval),
            sae.EvalNode("Affichage nom", 1.0, eval_func= sae.manualEval),
            sae.EvalNode("Affichage valeur", 1.0, eval_func= sae.manualEval),
            sae.EvalNode("Ordre des couches", 2.0, eval_func= sae.manualEval),
            sae.EvalNode("Gestion visibilité", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Empty initially:\n"
                                                     "Load 2+ slices -> appears\n"
                                                     "Load 1 slice -> disappear"))
        ])
    def informationTree(self):
        return sae.EvalNode("Affichage d'informations",
                            set_up_func= lambda : print("Load sparse"),
                            children=[
                                sae.EvalNode("getFrameMinMax", 1.0,
                                             eval_func= sae.manualEval,
                                             set_up_func= lambda : print("on slice 36: [-3024,344]")),
                                sae.EvalNode("getCollectionMinMax", 1.0,
                                             eval_func= sae.manualEval,
                                             set_up_func= lambda : print("Expecting [-3024,4043]"))
                            ])


    def get3DTree(self):
        return sae.EvalNode("Fonctionnalités 3D", children=[
            self.generating3DTree(),
            self.display3DTree(),
            self.cameraTree()
        ])

    def generating3DTree(self):
        return sae.EvalNode("Génération des points", children=[
            sae.EvalNode("Export des points", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("getOutputData rediriger vers struct")),
            sae.EvalNode("Utilisation 'window'", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("setWindow avant getOutputData")),
            sae.EvalNode("Stockage données", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Un seul tableau")),
            sae.EvalNode("Gestion dimensions", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("PixelWidth,PixelHeight,SliceThickness"))
        ])

    def display3DTree(self):
        return sae.EvalNode("Affichage 3D", children=[
            sae.EvalNode("Zone OpenGL", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Présence d'une zone OpenGL")),
            sae.EvalNode("Affichage de points", 2.0, eval_func= sae.manualEval),
            sae.EvalNode("Objet centré", 1.0, eval_func= sae.manualEval),
            sae.EvalNode("Transparence dynamique", 1.0, eval_func= sae.manualEval),
            sae.EvalNode("Masquage points vide", 1.0, eval_func= sae.manualEval,
                         set_up_func=lambda : print("Regarder dans le code")),
        ])

    def cameraTree(self):
        return sae.EvalNode("Caméra dynamique", children=[
            sae.EvalNode("Rotation possible", 1.0, eval_func= sae.manualEval),
            sae.EvalNode("Rotation ergonomique", 1.0, eval_func= sae.manualEval),
            sae.EvalNode("Zoom", 1.0, eval_func= sae.manualEval)
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
        dicom_eval = Dicom3DViewerAssignment(args.path, g)
        root = dicom_eval.run(original_evaluation)
        root.exportToJson(json_path)
        txt_content = sae.evalToString(root)
        txt_path = join(args.path, g.getKey() + ".txt")
        with open(txt_path, "w") as f:
            f.write(txt_content)
        print(txt_content)

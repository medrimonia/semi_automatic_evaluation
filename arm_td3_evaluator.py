#!/usr/bin/env python3

import argparse
import os
from os.path import join, dirname
from anytree.importer import JsonImporter

import semi_automatic_evaluator as sae

class PointCloudAssignment(sae.EvaluationProcess):
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
        return sae.EvalNode("Rendu 3", children=[
            self.getDefaultRulesTree(),
            self.get3DTree(),
            self.getPCLTree()
        ])

    def get3DTree(self):
        return sae.EvalNode("Fin TD: 3D Viewer", children=[
            self.customizableOptionsTree(),
            self.layerManagementTree(),
            self.widgetDisplayTree()
        ])

    def customizableOptionsTree(self):
        return sae.EvalNode("Gestions des options", children=[
            sae.EvalNode("Masquage des points vides", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option associée")),
            sae.EvalNode("Choix caméra", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Retour sur la même position?"))
        ])

    def layerManagementTree(self):
        return sae.EvalNode("Gestions des différentes couches", children=[
            sae.EvalNode("Mise en évidence", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option associée")),
            sae.EvalNode("Masquer couches précédentes", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option associée")),
            sae.EvalNode("Masquer couches suivantes", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option associée"))
        ])

    def widgetDisplayTree(self):
        return sae.EvalNode("Visibilité des widgets", children=[
            sae.EvalNode("Visibilité widget 2D", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option associée")),
            sae.EvalNode("Visibilité widget 3D", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option associée")),
            sae.EvalNode("Widget 3D caché et charge CPU", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Mise à jour Window 'fluide'")),
            sae.EvalNode("Redimensionnement", 1.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option associée"))
        ])

    def getPCLTree(self):
        return sae.EvalNode("TD: Point Cloud", children=[
            sae.EvalNode("Chargement des données", 2.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Voir fichier dicom_viewer.cpp et volumic_data.cpp")),
            sae.EvalNode("Fenêtre dynamique", 2.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option associée")),
            sae.EvalNode("Segmentation naive", 2.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester l'option avec différents nombre de couleurs")),
            sae.EvalNode("Filtrage des points internes", 2.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Option et disparition des points")),
            sae.EvalNode("Export PC", 2.0, eval_func= sae.manualEval,
                         set_up_func= lambda : print("Tester option et contenu associé (sans vérif meshlab)"))
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
        dicom_eval = PointCloudAssignment(args.path, g)
        root = dicom_eval.run(original_evaluation)
        root.exportToJson(json_path)
        txt_content = sae.evalToString(root)
        txt_path = join(args.path, g.getKey() + ".txt")
        with open(txt_path, "w") as f:
            f.write(txt_content)
        print(txt_content)

#!/usr/bin/env python3

import argparse
import os
import sys
from os.path import join, dirname
from anytree.importer import JsonImporter

import semi_automatic_evaluator as sae
import numpy as np
from scipy.spatial.transform import Rotation as R

import unittest

from importlib import reload, import_module

rtol = 1e-3
atol = 1e-3

ht_module = None
ctrl_module = None

class MGIAssignment(sae.EvaluationProcess):
    def __init__(self, path, group):
        super().__init__(path, group)

    def _eval(self):
        global ht_module, ctrl_module
        # Eval rules respect even if modules are not found
        # self.root.children[0].eval()
        current_dir = os.getcwd()
        assignment_dir = join(current_dir,dirname(self._archive_path))
        group_key = self._group.getKey()
        self.group_path = os.path.join(os.path.dirname(self._archive_path), group_key)
        sys.path.append(join(current_dir, self.group_path))
        if ht_module is not None:
            ht_module = reload(ht_module)
            ctrl_module = reload(ctrl_module)
        else:
            ht_module = import_module("homogeneous_transform")
            ctrl_module = import_module("control")
        sys.path.pop()
        print(ht_module)
        print(ctrl_module)
        self.rt_robot = getattr(ctrl_module, "RTRobot")
        self.rrr_robot = getattr(ctrl_module, "RRRRobot")
        self.leg_robot = getattr(ctrl_module, "LegRobot")
        for c in self.root.children[1:]:
            print("Evaluating: " + c.name)
            c.eval()

    def getStructure(self):
        return sae.EvalNode("2-MGI et Jacobienne", children=[
            self.getDefaultRulesTree(),
            self.getRobotTree("rt"),
            # self.getRobotTree("RRR"),
            # self.getRobotTree("Leg"),
            # sae.EvalNode("Compte rendu", eval_func=sae.manualEval, max_points=3)
            ])

    def getRobotTree(self, robot_name):
        robot_node = sae.EvalNode("Robot {:}".format(robot_name),
                                  set_up_func= lambda : self.initializeRobot(robot_name))
        dic = {"Jacobienne" : "jacobian", "MGI Analytique" : "analytical_mgi"}
        for test_name, method_name in dic.items():
            print("{:} -> {:}".format(test_name, method_name))
            pattern = "{:}_robot_{:}".format(robot_name,method_name)
            methods = [m for m in dir(self) if pattern in m]
            if (len(methods) == 0):
                continue
            test_points = 1
            if test_name == "MGI Analytique" :
                test_points = 2
            case_points = test_points / len(methods)
            test_node = sae.EvalNode(test_name, parent= robot_node)
            for m in methods:
                print(m)
                callback = getattr(self,m)
                print(callback)
                sae.EvalNode(m[m.rfind('_')+1:],
                             eval_func = lambda node, func=getattr(self,m) : sae.assertionEval(
                                 node, func),
                             max_points = case_points,
                             parent = test_node)
        return robot_node

    def fail(self):
        raise RuntimeError("error")

    def execute(self, method):
        print("Executing {:}".format(method))
        getattr(self,method)()

    def initializeRobot(self,robot_name):
        if robot_name == "rt":
            self.robot = self.rt_robot()
        else:
            raise RuntimeError("unexpected robot name {:}".format(robot_name))

    def test_rt_robot_jacobian_config0(self):
        received = self.robot.computeJacobian(np.array([0,0]))
        expected = np.array([[0.25,0.2],[1.0,0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rt_robot_jacobian_config1(self):
        # config: [pi/2, 0]
        received = self.robot.computeJacobian(np.array([np.pi/2,0]))
        expected = np.array([[-0.2,0.25],[0.0,1.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rt_robot_jacobian_config2(self):
        # config: [0, 0.1]
        received = self.robot.computeJacobian(np.array([0,0.1]))
        expected = np.array([[0.25,0.3],[1.0,0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rt_robot_analytical_mgi_config0(self):
        nb_sol, sol = self.robot.analyticalMGI(np.array([0.2,-0.25]))
        expected_sol = np.array([0,0])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_rt_robot_analytical_mgi_config1(self):
        # config: [0, 0.2]
        nb_sol, sol = self.robot.analyticalMGI(np.array([0.4,-0.25]))
        expected_sol = np.array([0,0.2])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_rt_robot_analytical_mgi_config2(self):
        # config: [np.pi/2, 0.1]
        nb_sol, sol = self.robot.analyticalMGI(np.array([0.25,0.3]))
        expected_sol = np.array([np.pi/2,0.1])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_rt_robot_analytical_mgi_too_far(self):
        # unreachable config
        nb_sol, sol = self.robot.analyticalMGI(np.array([0.46,-0.25]))
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)

# IMPLEMENTING DEFAULT VERSION TO TEST BEHAVIORS
# ----------------------------------------------

class TestRT(unittest.TestCase):
    def __init__(self, module):
        super()

    @classmethod
    def setUp(self):
        self.model = RTRobot()


    def test_rt_robot_analytical_mgi_0(self):
        nb_sol, sol = self.model.analyticalMGI(np.array([0.2,-0.25]))
        expected_sol = np.array([0,0])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_rt_robot_analytical_mgi_config1(self):
        # config: [0, 0.2]
        nb_sol, sol = self.model.analyticalMGI(np.array([0.4,-0.25]))
        expected_sol = np.array([0,0.2])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_rt_robot_analytical_mgi_config2(self):
        # config: [np.pi/2, 0.1]
        nb_sol, sol = self.model.analyticalMGI(np.array([0.25,0.3]))
        expected_sol = np.array([np.pi/2,0.1])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_rt_robot_analytical_mgi_too_far(self):
        # unreachable config
        nb_sol, sol = self.model.analyticalMGI(np.array([0.46,-0.25]))
        expected_sol = np.array([np.pi/2,0.1])
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)

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
        # if os.path.isfile(json_path):
        #     print("Importing existing evaluation for the group: " + g.getKey())
        #     with open(json_path, 'r') as f:
        #         original_evaluation = JsonImporter().read(f)
        dicom_eval = MGIAssignment(args.path, g)
        root = dicom_eval.run(original_evaluation)
        root.exportToJson(json_path)
        txt_content = sae.evalToString(root)
        txt_path = join(args.path, g.getKey() + ".txt")
        with open(txt_path, "w") as f:
            f.write(txt_content)
        print(txt_content)

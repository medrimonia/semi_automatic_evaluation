#!/usr/bin/env python3

import argparse
import os
import sys
import math
from os.path import join, dirname
from anytree.importer import JsonImporter

import semi_automatic_evaluator as sae
import numpy as np
from scipy.spatial.transform import Rotation as R

import unittest

from importlib import reload, import_module

rtol = 1e-3
atol = 1e-2

ht_module = None
ctrl_module = None

def iterativeTest(robot, initial_pos, target, method, nb_iterations = 1):
    # Nb iterations can be used to ensure long-term convergence
    joints = initial_pos
    for i in range(nb_iterations):
        joints = robot.computeMGI(joints, target, method)
    final_pos = robot.computeMGD(joints)
    np.testing.assert_allclose(target, final_pos, rtol, atol)

class MGIAssignment(sae.EvaluationProcess):
    def __init__(self, path, group):
        super().__init__(path, group)

    def _eval(self):
        global ht_module, ctrl_module
        # Eval rules respect even if modules are not found
        self.root.children[0].eval()
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
            self.getRobotTree("rrr"),
            self.getRobotTree("leg"),
            sae.EvalNode("Compte rendu", eval_func=sae.manualEval, max_points=3)
            ])

    def getRobotTree(self, robot_name):
        robot_node = sae.EvalNode("Robot {:}".format(robot_name),
                                  set_up_func= lambda robot_name=robot_name : self.initializeModel(robot_name))
        # Adding limits test
        limit_test_name = "test_{:}_operational_limits".format(robot_name)
        sae.EvalNode("Limites espace opérationnel",
                     eval_func = lambda node, f=getattr(self,limit_test_name): sae.assertionEval(node, f),
                     parent = robot_node)
        # Adding all other tests
        dic = {
            "Jacobienne" : "jacobian",
            "Jacobienne inverse" : "jac_inverse",
            "Jacobienne transposée" : "jac_transposed",
            "MGI Analytique" : "analytical_mgi"}
        for test_name, method_name in dic.items():
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
                callback = getattr(self,m)
                sae.EvalNode(m[m.rfind('_')+1:],
                             eval_func = lambda node, func=getattr(self,m) : sae.assertionEval(
                                 node, func),
                             max_points = case_points,
                             parent = test_node)
        return robot_node

    def initializeModel(self,robot_name):
        if robot_name == "rt":
            self.model = self.rt_robot()
        elif robot_name == "rrr":
            self.model = self.rrr_robot()
        elif robot_name == "leg":
            self.model = self.leg_robot()
        else:
            raise RuntimeError("unexpected robot name {:}".format(robot_name))

    def test_rt_operational_limits(self):
        D = math.sqrt((0.2+0.25)**2 + 0.25**2)
        expected = [[-D,D],[-D,D]]
        received = self.model.getOperationalDimensionLimits()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rt_robot_jacobian_config0(self):
        received = self.model.computeJacobian(np.array([0,0]))
        expected = np.array([[0.25,0.2],[1.0,0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rt_robot_jacobian_config1(self):
        # config: [pi/2, 0]
        received = self.model.computeJacobian(np.array([np.pi/2,0]))
        expected = np.array([[-0.2,0.25],[0.0,1.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rt_robot_jacobian_config2(self):
        # config: [0, 0.1]
        received = self.model.computeJacobian(np.array([0,0.1]))
        expected = np.array([[0.25,0.3],[1.0,0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rt_robot_analytical_mgi_config0(self):
        nb_sol, sol = self.model.analyticalMGI(np.array([0.2,-0.25]))
        expected_sol = np.array([0,0])
        np.testing.assert_equal(nb_sol, 1)
        np.testing.assert_allclose(sol, expected_sol, rtol, atol)

    def test_rt_robot_jac_inverse_long0(self):
        iterativeTest(self.model, np.array([0,0]), np.array([0.25,0.3]), "jacobianInverse", 50)

    def test_rt_robot_jac_inverse_long1(self):
        # This test fails because the search starts alternating between two
        # different configurations. There are no easy apparent way to fix this.
        iterativeTest(self.model, np.array([0,0]), np.array([-0.2,0.25]), "jacobianInverse", 50)

    def test_rt_robot_jac_inverse_short(self):
        iterativeTest(self.model, np.array([0,0]), np.array([0.25,0.3]), "jacobianInverse", 1)

    def test_rt_robot_jac_transposed_config0(self):
        iterativeTest(self.model, np.array([0,0]), np.array([0.25,0.3]), "jacobianTransposed", 1)

    def test_rt_robot_jac_transposed_config1(self):
        # This test fails because the search starts alternating between two
        # different configurations. There are no easy apparent way to fix this.
        iterativeTest(self.model, np.array([0,0]), np.array([-0.2,0.25]), "jacobianTransposed", 1)

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
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)

    def test_rrr_operational_limits(self):
        D1 = 0.3 + 0.31
        D2 = 0.4 + 0.3 + 0.31
        Z = 1.01
        expected = [[-D2,D2],[-D2,D2],[Z-D1,Z+D1]]
        received = self.model.getOperationalDimensionLimits()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rrr_robot_jacobian_config0(self):
        received = self.model.computeJacobian(np.array([0,0,0]))
        expected = np.array([[-1.01,0.0,0.0],[0.0,0.0,0.61],[0.0,0.0,0.31]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rrr_robot_jacobian_config1(self):
        # config: [pi/2, 0, 0]
        received = self.model.computeJacobian(np.array([np.pi/2,0,0]))
        expected = np.array([[0.0,-1.01,0.0],[0.0,0.0,0.61],[0.0,0.0,0.31]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rrr_robot_jacobian_config2(self):
        # config: [0, pi/2, 0]
        received = self.model.computeJacobian(np.array([0,np.pi/2,0]))
        expected = np.array([[-0.4,0.0,0.0],[0.0,-0.61,0.0],[0.0,-0.31,0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rrr_robot_jacobian_config3(self):
        # config: [0, 0, pi/2]
        received = self.model.computeJacobian(np.array([0,0,np.pi/2]))
        expected = np.array([[-0.7,0.0,0.0],[0.0,-0.31,0.3],[0.0,-0.31,0.0]]).transpose()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rrr_robot_jac_inverse_long1(self):
        iterativeTest(self.model, np.array([0,0.1,0]), np.array([0.0,0.5,1.01]), "jacobianInverse", 50)

    def test_rrr_robot_jac_inverse_long2(self):
        iterativeTest(self.model, np.array([0,0.2,0]), np.array([0.3,-0.5,1.2]), "jacobianInverse", 50)

    def test_rrr_robot_jac_inverse_short(self):
        iterativeTest(self.model, np.array([0,0.1,0]), np.array([0.0,0.5,1.01]), "jacobianInverse", 1)

    def test_rrr_robot_jac_inverse_singularity(self):
        iterativeTest(self.model, np.array([0,0,0]), np.array([0.0,-0.5,1.01]), "jacobianInverse", 50)

    def test_rrr_robot_jac_transposed_config1(self):
        iterativeTest(self.model, np.array([0,0.1,0]), np.array([0.0,0.6,1.01]), "jacobianTransposed", 50)

    def test_rrr_robot_jac_transposed_config2(self):
        iterativeTest(self.model, np.array([0,0.2,0]), np.array([0.6,-0.2,1.2]), "jacobianTransposed", 50)

    def test_rrr_robot_jac_transposed_singularity(self):
        iterativeTest(self.model, np.array([0,0,0]), np.array([0.0,-0.5,1.01]), "jacobianTransposed", 1)

    def test_rrr_robot_analytical_mgi_config0(self):
        # Adding a tiny offset to make 'sure' that target is considered as
        # reachable despite floating point issues
        operational_pos = np.array([0.0,1.01-10**-14,1.01])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 2)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_rrr_robot_analytical_mgi_config1(self):
        # Adding a tiny offset to make 'sure' that target is considered as
        # reachable despite floating point issues
        operational_pos = np.array([-0.4,0.0,1.01 + 0.3 + 0.31-10**-14])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 2)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_rrr_robot_analytical_mgi_config2(self):
        # No need for offset here, not at the border of the reachable space
        operational_pos = np.array([0.71, 0.0,1.01 + 0.3])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 2)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_rrr_robot_analytical_mgi_config3(self):
        # Classic exemple near center, 4 solutions expected
        # No need for offset here, not at the border of the reachable space
        operational_pos = np.array([0.05,0.05,1.05])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 4)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_rrr_robot_analytical_mgi_singularity(self):
        # Exemple at center, depending on floating points approximations, answer
        # might be -1 or 4
        operational_pos = np.array([0.0,0.0,1.05])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        received_pos = self.model.computeMGD(sol)
        np.testing.assert_equal(nb_sol, -1)
        np.testing.assert_allclose(operational_pos, received_pos, rtol, atol)

    def test_rrr_robot_analytical_mgi_unreachable1(self):
        # target too far
        operational_pos = np.array([0.0,1.05,1.05])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)

    def test_rrr_robot_analytical_mgi_unreachable2(self):
        # target too far
        operational_pos = np.array([0.0,0.05,2.0])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)

    def test_rrr_robot_analytical_mgi_unreachable3(self):
        # target too close to 'q1' position
        operational_pos = np.array([0.0,0.4,1.01])
        nb_sol, sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sol, 0)
        np.testing.assert_equal(sol, None)

    def test_leg_operational_limits(self):
        D1 = math.sqrt((0.3 + 0.3 + 0.2)**2 + 0.02**2)
        D2 = math.sqrt((0.4 + 0.3 + 0.3 + 0.2)**2 + 0.02**2)
        Z = 1.01
        expected = [[-D2,D2],[-D2,D2],[Z-D1,Z+D1],[-1,1]]
        received = self.model.getOperationalDimensionLimits()
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_compute_mgd_config0(self):
        joints = np.array([0,0,0,0])
        expected = np.array([0.02,1.2,1.01,0.0])
        received = self.model.computeMGD(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_compute_mgd_config1(self):
        joints = np.array([0,0,0,-np.pi/2])
        expected = np.array([0.02,1.0,0.81,-1.0])
        received = self.model.computeMGD(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_compute_mgd_config2(self):
        joints = np.array([-np.pi/2,np.pi/2,-np.pi/2,np.pi/2])
        expected = np.array([0.7,-0.02,1.51,1.0])
        received = self.model.computeMGD(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_jacobian_config0(self):
        joints = np.array([0,0,0,0])
        expected = np.array(
            [
                [-1.2,0,0,0],
                [0.02,0.0,0.0,0.0],
                [0.0,0.8,0.5,0.2],
                [0.0,1.0,1.0,1.0]
            ])
        received = self.model.computeJacobian(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_jacobian_config1(self):
        # Initial direction: x+
        # At q[2], starts pointing down
        joints = np.array([-np.pi/2,0,-np.pi/2,0])
        expected = np.array(
            [
                [0.02,0.5,0.5,0.2],
                [0.7,0.0,0.0,0.0],
                [0.0,0.3,0.0,0.0],
                [0.0,0.0,0.0,0.0]
            ])
        received = self.model.computeJacobian(joints)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_leg_robot_jac_inverse_long0(self):
        iterativeTest(self.model,
                      np.array([0,0.1,0,0]),
                      np.array([0.0,0.7,0.8,-1.0]),
                      "jacobianInverse", 50)

    def test_leg_robot_jac_inverse_long1(self):
        iterativeTest(self.model,
                      np.array([0,0.1,0,0]),
                      np.array([0.3,0.5,1.2,0.5]),
                      "jacobianInverse", 50)

    def test_leg_robot_jac_inverse_short(self):
        iterativeTest(self.model,
                      np.array([0,0.1,0,0]),
                      np.array([0.0,0.7,0.8,-1.0]),
                      "jacobianInverse", 1)

    def test_leg_robot_jac_transposed_long0(self):
        iterativeTest(self.model,
                      np.array([0,0.1,0,0]),
                      np.array([0.0,0.7,0.8,-1.0]),
                      "jacobianTransposed", 50)

    def test_leg_robot_jac_transposed_long1(self):
        iterativeTest(self.model,
                      np.array([0,0.1,0,0]),
                      np.array([0.3,0.5,1.2,0.5]),
                      "jacobianTransposed", 50)

    def test_leg_robot_analytical_mgi_config0(self):
        # Adding small offset to ensure we're in a reachable position
        operational_pos = np.array([0.02,1.2 - 1e-7,1.01,0.0])
        expected_sol = np.array([0,0,0,0])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sols > 0, True)
        np.testing.assert_allclose(received_sol, expected_sol, rtol, atol)

    def test_leg_robot_analytical_mgi_config1(self):
        operational_pos = np.array([0.2,0.5,0.9,-0.5])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        # Here we have 2 solutions for q123 and for each of those we have two
        # solutions for q12
        np.testing.assert_equal(nb_sols, 4)
        backward_pos = self.model.computeMGD(received_sol)
        np.testing.assert_allclose(backward_pos, operational_pos, rtol, atol)

    def test_leg_robot_analytical_mgi_config2(self):
        operational_pos = np.array([0.0,0.2,0.8,-0.9])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        # Here, there is one option where config can be reached with the non-obvious
        # position for first angle
        np.testing.assert_equal(nb_sols, 6)
        backward_pos = self.model.computeMGD(received_sol)
        np.testing.assert_allclose(backward_pos, operational_pos, rtol, atol)

    def test_leg_robot_analytical_mgi_invalid_unreachable1(self):
        # Due to the link_offset, position can't be reached
        operational_pos = np.array([0.0,0.01,0.8,-0.9])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sols, 0)

    def test_leg_robot_analytical_mgi_invalid_unreachable2(self):
        # A position clearly too far to be reached
        operational_pos = np.array([0.0,1.4,1.0,0.0])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sols, 0)

    def test_leg_robot_analytical_mgi_invalid_unreachable3(self):
        # Due to the link_offset, position can't be reached
        operational_pos = np.array([0.0,1.1,1.0,-1.0])
        nb_sols, received_sol = self.model.analyticalMGI(operational_pos)
        np.testing.assert_equal(nb_sols, 0)

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
        dicom_eval = MGIAssignment(args.path, g)
        root = dicom_eval.run(original_evaluation)
        root.exportToJson(json_path)
        txt_content = sae.evalToString(root)
        txt_path = join(args.path, g.getKey() + ".txt")
        with open(txt_path, "w") as f:
            f.write(txt_content)
        print(txt_content)

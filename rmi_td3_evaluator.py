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
traj_module = None

class TrajectoriesAssignment(sae.EvaluationProcess):
    def __init__(self, path, group):
        super().__init__(path, group)

    def _eval(self):
        global ht_module, ctrl_module, traj_module
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
            traj_module = reload(traj_module)
        else:
            ht_module = import_module("homogeneous_transform")
            ctrl_module = import_module("control")
            traj_module = import_module("trajectories")
        sys.path.pop()
        print(ht_module)
        print(ctrl_module)
        print(traj_module)
        self.traj_builder = getattr(traj_module, "buildTrajectory")
        print(self.traj_builder)
        for c in self.root.children[1:]:
            print("Evaluating: " + c.name)
            c.eval()

    def getStructure(self):
        return sae.EvalNode("3-Trajectoires", children=[
            self.getDefaultRulesTree(),
            self.get1DTrajectoriesTree()
            # self.getRobotTrajectoriesTree(),
            # sae.EvalNode("Compte rendu", eval_func=sae.manualEval, max_points=3)
            ])

    def get1DTrajectoriesTree(self):
        trajectories = [
            "ConstantSpline",
            "LinearSpline",
            "CubicZeroDerivativeSpline",
            "CubicWideStencilSpline",
            "CubicCustomDerivativeSpline",
            "NaturalCubicSpline",
            "PeriodicCubicSpline",
            "TrapezoidalVelocity"
        ]
        node = sae.EvalNode("1D Trajectories",
                            children = [self.getTrajectoryTree(traj_name)
                                        for traj_name in trajectories])
        return node

    def getTrajectoryTree(self, traj_name):
        traj_node = sae.EvalNode(traj_name)
        dic = {
            "Degré" : "degree",
            "Constructeur invalide" : "invalid",
            "Limites" : "limits",
            "Position" : "position",
            "Vitesse" : "vel",
            "Acceleration" : "acc"
        }
        for test_name, method_name in dic.items():
            pattern = "test_{:}_{:}".format(traj_name, method_name)
            methods = [m for m in dir(self) if pattern in m]
            if (len(methods) == 0):
                continue
            test_points = 0.2
            if test_name == "Position" :
                test_points = 0.5
            if len(methods) == 1:
                sae.EvalNode(test_name,
                             eval_func = lambda node, func=getattr(self,methods[0]) : sae.assertionEval(
                                 node, func),
                             max_points = test_points,
                             parent = traj_node)
            else:
                case_points = test_points / len(methods)
                test_node = sae.EvalNode(test_name, parent=traj_node)
                for m in methods:
                    callback = getattr(self,m)
                    sae.EvalNode(m[m.rfind('_')+1:],
                                 eval_func = lambda node, func=getattr(self,m) : sae.assertionEval(
                                     node, func),
                                 max_points = case_points,
                                 parent = test_node)
        return traj_node

    def getDefaultConstantSpline(self):
        return self.traj_builder("ConstantSpline", 0.0,
                                 np.array([[0,2],[1,3],[2,-2]]))

    def getUnorderedConstantSpline(self):
        return self.traj_builder("ConstantSpline", 0.0,
                                 np.array([[0,2],[3,3],[2,-2]]))

    def test_ConstantSpline_degree(self):
        traj = self.getDefaultConstantSpline()
        np.testing.assert_equal(0, traj.getDegree())

    def test_ConstantSpline_limits_before(self):
        traj = self.getDefaultConstantSpline()
        np.testing.assert_equal(2, traj.getVal(-1,0))
        np.testing.assert_equal(0, traj.getVal(-1,1))
        np.testing.assert_equal(0, traj.getVal(-1,2))
    def test_ConstantSpline_limits_after(self):
        traj = self.getDefaultConstantSpline()
        np.testing.assert_equal(-2, traj.getVal(3,0))
        np.testing.assert_equal(0, traj.getVal(3,1))
        np.testing.assert_equal(0, traj.getVal(3,2))
    def test_ConstantSpline_position_controlPoints(self):
        traj = self.getDefaultConstantSpline()
        np.testing.assert_equal( 2, traj.getVal(0, 0))
        np.testing.assert_equal( 3, traj.getVal(1, 0))
        np.testing.assert_equal(-2, traj.getVal(2, 0))
    def test_ConstantSpline_position_intermediary(self):
        traj = self.getDefaultConstantSpline()
        np.testing.assert_equal(2, traj.getVal(0.25, 0))
        np.testing.assert_equal(2, traj.getVal(0.75, 0))
        np.testing.assert_equal(3, traj.getVal(1.25, 0))
        np.testing.assert_equal(3, traj.getVal(1.75, 0))
    def test_ConstantSpline_vel(self):
        traj = self.getDefaultConstantSpline()
        np.testing.assert_equal(0, traj.getVal(0.33, 1))
        np.testing.assert_equal(0, traj.getVal(1.53, 1))

    def test_ConstantSpline_acc(self):
        traj = self.getDefaultConstantSpline()
        np.testing.assert_equal(0, traj.getVal(0.33, 2))
        np.testing.assert_equal(0, traj.getVal(1.53, 2))

    def test_ConstantSpline_invalid_unorderedPoints(self):
        with np.testing.assert_raises(Exception):
            self.getUnorderedConstantSpline()

    def getDefaultLinearSpline(self):
        return self.traj_builder("LinearSpline", 0.0,
                                 np.array([[0,2],[1,3],[3,-3]]))

    def getUnorderedLinearSpline(self):
        return self.traj_builder("LinearSpline", 0.0,
                                 np.array([[0,2],[3,3],[2,-2]]))

    def test_LinearSpline_degree(self):
        traj = self.getDefaultLinearSpline()
        np.testing.assert_equal(1, traj.getDegree())

    def test_LinearSpline_limits_before(self):
        traj = self.getDefaultLinearSpline()
        np.testing.assert_equal(2, traj.getVal(-1,0))
        np.testing.assert_equal(0, traj.getVal(-1,1))
        np.testing.assert_equal(0, traj.getVal(-1,2))
    def test_LinearSpline_limits_after(self):
        traj = self.getDefaultLinearSpline()
        np.testing.assert_equal(-3, traj.getVal(4,0))
        np.testing.assert_equal( 0, traj.getVal(4,1))
        np.testing.assert_equal( 0, traj.getVal(4,2))
    def test_LinearSpline_position_controlPoints(self):
        traj = self.getDefaultLinearSpline()
        np.testing.assert_equal( 2, traj.getVal(0, 0))
        np.testing.assert_equal( 3, traj.getVal(1, 0))
        np.testing.assert_equal(-3, traj.getVal(3, 0))
    def test_LinearSpline_position_intermediary(self):
        traj = self.getDefaultLinearSpline()
        np.testing.assert_equal( 2.25, traj.getVal(0.25, 0))
        np.testing.assert_equal( 2.75, traj.getVal(0.75, 0))
        np.testing.assert_equal( 1.50, traj.getVal(1.50, 0))
        np.testing.assert_equal(-1.50, traj.getVal(2.50, 0))
    def test_LinearSpline_vel(self):
        traj = self.getDefaultLinearSpline()
        np.testing.assert_equal( 1, traj.getVal(0.33, 1))
        np.testing.assert_equal(-3, traj.getVal(1.53, 1))
    def test_LinearSpline_acc(self):
        traj = self.getDefaultLinearSpline()
        np.testing.assert_equal(0, traj.getVal(0.33, 2))
        np.testing.assert_equal(0, traj.getVal(1.53, 2))
    def test_LinearSpline_invalid_unorderedPoints(self):
        with np.testing.assert_raises(Exception):
            self.getUnorderedLinearSpline()

    def getDefaultCubicZeroDerivativeSpline(self):
        return self.traj_builder("CubicZeroDerivativeSpline", 0.0,
                                 np.array([[0,2],[1,3],[3,-3]]))
    def getUnorderedCubicZeroDerivativeSpline(self):
        return self.traj_builder("CubicZeroDerivativeSpline", 0.0,
                                 np.array([[0,2],[3,3],[2,-2]]))
    def test_CubicZeroDerivativeSpline_degree(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_equal(3, traj.getDegree())

    def test_CubicZeroDerivativeSpline_limits_before(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_equal(2, traj.getVal(-1,0))
        np.testing.assert_equal(0, traj.getVal(-1,1))
        np.testing.assert_equal(0, traj.getVal(-1,2))
    def test_CubicZeroDerivativeSpline_limits_after(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_equal(-3, traj.getVal(4,0))
        np.testing.assert_equal( 0, traj.getVal(4,1))
        np.testing.assert_equal( 0, traj.getVal(4,2))
    def test_CubicZeroDerivativeSpline_position_controlPoints(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_equal( 2, traj.getVal(0))
        np.testing.assert_equal( 3, traj.getVal(1))
        np.testing.assert_equal(-3, traj.getVal(3))
    def test_CubicZeroDerivativeSpline_position_intermediary(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_allclose( 2.5, traj.getVal(0.5), rtol, atol)
        np.testing.assert_allclose( 0.0, traj.getVal(2.0), rtol, atol)
    def test_CubicZeroDerivativeSpline_vel_controlPoints(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_allclose(0, traj.getVal(0, 1), rtol, atol)
        np.testing.assert_allclose(0, traj.getVal(0, 1), rtol, atol)
    def test_CubicZeroDerivativeSpline_vel_intermediary(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_allclose( 1.125, traj.getVal(0.25, 1), rtol, atol)
        np.testing.assert_allclose( 1.125, traj.getVal(0.75, 1), rtol, atol)
        np.testing.assert_allclose(-3.375, traj.getVal(1.50, 1), rtol, atol)
        np.testing.assert_allclose(-3.375, traj.getVal(2.50, 1), rtol, atol)
    def test_CubicZeroDerivativeSpline_acc(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_allclose(6, traj.getVal(0.0, 2), rtol, atol)
        np.testing.assert_allclose(0, traj.getVal(0.5, 2), rtol, atol)
    def test_CubicZeroDerivativeSpline_invalid_unorderedPoints(self):
        with np.testing.assert_raises(Exception):
            self.getUnorderedCubicZeroDerivativeSpline()
    def getDefaultCubicWideStencilSpline(self):
        return self.traj_builder("CubicWideStencilSpline", 0.0,
                                 np.array([[0,2],[1,3],[2,1],[3,-3],[5,-2]]))
    def getTooShortCubicWideStencilSpline(self):
        return self.traj_builder("CubicWideStencilSpline", 0.0,
                                 np.array([[0,2],[1,3],[2,-2]]))
    def test_CubicWideStencilSpline_degree(self):
        traj = self.getDefaultCubicWideStencilSpline()
        np.testing.assert_equal(3, traj.getDegree())

    def test_CubicWideStencilSpline_limits_before(self):
        traj = self.getDefaultCubicWideStencilSpline()
        np.testing.assert_equal(2, traj.getVal(-1,0))
        np.testing.assert_equal(0, traj.getVal(-1,1))
        np.testing.assert_equal(0, traj.getVal(-1,2))
    def test_CubicWideStencilSpline_limits_after(self):
        traj = self.getDefaultCubicWideStencilSpline()
        np.testing.assert_equal(-2, traj.getVal(6,0))
        np.testing.assert_equal( 0, traj.getVal(6,1))
        np.testing.assert_equal( 0, traj.getVal(6,2))
    def test_CubicWideStencilSpline_position_controlPoints(self):
        traj = self.getDefaultCubicWideStencilSpline()
        np.testing.assert_allclose( 2, traj.getVal(0), rtol, atol)
        np.testing.assert_allclose( 3, traj.getVal(1), rtol, atol)
        np.testing.assert_allclose( 1, traj.getVal(2), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(3), rtol, atol)
        np.testing.assert_allclose(-2, traj.getVal(5), rtol, atol)
    def test_CubicWideStencilSpline_position_intermediary(self):
        traj = self.getDefaultCubicWideStencilSpline()
        np.testing.assert_allclose( 2.9375, traj.getVal(0.5, 0), rtol, atol)
        np.testing.assert_allclose(-0.984, traj.getVal(2.5, 0), rtol, atol)
    def test_CubicWideStencilSpline_vel(self):
        traj = self.getDefaultCubicWideStencilSpline()
        np.testing.assert_allclose( 2.833, traj.getVal(0, 1), rtol, atol)
        np.testing.assert_allclose(-4.156, traj.getVal(2.5, 1), rtol, atol)
    def test_CubicWideStencilSpline_acc(self):
        traj = self.getDefaultCubicWideStencilSpline()
        np.testing.assert_allclose(-4.000, traj.getVal(0.0, 2), rtol, atol)
        np.testing.assert_allclose(-0.125, traj.getVal(2.5, 2), rtol, atol)
    def test_CubicWideStencilSpline_invalid_missingPoints(self):
        with np.testing.assert_raises(Exception):
            self.getTooShortCubicWideStencilSpline()
    def getDefaultCubicCustomDerivativeSpline(self):
        return self.traj_builder("CubicCustomDerivativeSpline", 0.0,
                                 np.array([[0,2,1],[1,3,-1],[3,-3,0]]))
    def getBadKnotsCubicCustomDerivativeSpline(self):
        return self.traj_builder("CubicCustomDerivativeSpline", 0.0,
                                 np.array([[0,2],[3,3],[2,-2]]))
    def test_CubicCustomDerivativeSpline_degree(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_equal(3, traj.getDegree())

    def test_CubicCustomDerivativeSpline_limits_before(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_equal(2, traj.getVal(-1,0))
        np.testing.assert_equal(0, traj.getVal(-1,1))
        np.testing.assert_equal(0, traj.getVal(-1,2))
    def test_CubicCustomDerivativeSpline_limits_after(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_equal(-3, traj.getVal(4,0))
        np.testing.assert_equal( 0, traj.getVal(4,1))
        np.testing.assert_equal( 0, traj.getVal(4,2))
    def test_CubicCustomDerivativeSpline_position_controlPoints(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_equal( 2, traj.getVal(0))
        np.testing.assert_equal( 3, traj.getVal(1))
        np.testing.assert_equal(-3, traj.getVal(3))
    def test_CubicCustomDerivativeSpline_position_intermediary(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_allclose( 2.75, traj.getVal(0.5), rtol, atol)
        np.testing.assert_allclose(-0.25, traj.getVal(2.0), rtol, atol)
    def test_CubicCustomDerivativeSpline_vel_controlPoints(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_allclose( 1, traj.getVal(0, 1), rtol, atol)
        np.testing.assert_allclose(-1, traj.getVal(1, 1), rtol, atol)
        np.testing.assert_allclose( 0, traj.getVal(3, 1), rtol, atol)
    def test_CubicCustomDerivativeSpline_vel_intermediary(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_allclose( 1.625, traj.getVal(0.25, 1), rtol, atol)
        np.testing.assert_allclose( 0.625, traj.getVal(0.75, 1), rtol, atol)
        np.testing.assert_allclose(-3.063, traj.getVal(2.50, 1), rtol, atol)
    def test_CubicCustomDerivativeSpline_acc(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_allclose( 4, traj.getVal(0.0, 2), rtol, atol)
        np.testing.assert_allclose(-2, traj.getVal(0.5, 2), rtol, atol)
    def test_CubicCustomDerivativeSpline_invalid_badKnots(self):
        with np.testing.assert_raises(Exception):
            self.getBadKnotsCubicCustomDerivativeSpline()

    def getDefaultNaturalCubicSpline(self):
        return self.traj_builder("NaturalCubicSpline", 0.0,
                                 np.array([[0,2],[1,3],[3,-3]]))
    def getUnorderedNaturalCubicSpline(self):
        return self.traj_builder("NaturalCubicSpline", 0.0,
                                 np.array([[0,2],[3,3],[2,-2]]))
    def test_NaturalCubicSpline_degree(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_equal(3, traj.getDegree())

    def test_NaturalCubicSpline_limits_before(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_equal(2, traj.getVal(-1,0))
        np.testing.assert_equal(0, traj.getVal(-1,1))
        np.testing.assert_equal(0, traj.getVal(-1,2))
    def test_NaturalCubicSpline_limits_after(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_equal(-3, traj.getVal(4,0))
        np.testing.assert_equal( 0, traj.getVal(4,1))
        np.testing.assert_equal( 0, traj.getVal(4,2))
    def test_NaturalCubicSpline_position_controlPoints(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_allclose( 2, traj.getVal(0), rtol, atol)
        np.testing.assert_allclose( 3, traj.getVal(1), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(3), rtol, atol)
    def test_NaturalCubicSpline_position_intermediary(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_allclose(2.75, traj.getVal(0.5), rtol, atol)
        np.testing.assert_allclose(1.00, traj.getVal(2.0), rtol, atol)
    def test_NaturalCubicSpline_vel(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_allclose( 1.542, traj.getVal(0.25, 1), rtol, atol)
        np.testing.assert_allclose(-4.083, traj.getVal(2.50, 1), rtol, atol)
    def test_NaturalCubicSpline_acc_border(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_allclose(0, traj.getVal(0.0, 2), rtol, atol)
        np.testing.assert_allclose(0, traj.getVal(3.0, 2), rtol, atol)
    def test_NaturalCubicSpline_acc_intermediary(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_allclose(-2.0, traj.getVal(0.5, 2), rtol, atol)
        np.testing.assert_allclose(-0.6, traj.getVal(2.7, 2), rtol, atol)
    def test_NaturalCubicSpline_invalid_unorderedPoints(self):
        with np.testing.assert_raises(Exception):
            self.getUnorderedNaturalCubicSpline()
    def getDefaultPeriodicCubicSpline(self):
        return self.traj_builder("PeriodicCubicSpline", 0.0,
                                 np.array([[0,2],[1,3],[3,-3],[5,2]]))
    def getBadLimitsPeriodicCubicSpline(self):
        return self.traj_builder("PeriodicCubicSpline", 0.0,
                                 np.array([[0,2],[3,3],[2,-2]]))
    def test_PeriodicCubicSpline_degree(self):
        traj = self.getDefaultPeriodicCubicSpline()
        np.testing.assert_equal(3, traj.getDegree())
    def test_PeriodicCubicSpline_position_controlPoints(self):
        traj = self.getDefaultPeriodicCubicSpline()
        np.testing.assert_allclose( 2, traj.getVal(0), rtol, atol)
        np.testing.assert_allclose( 3, traj.getVal(1), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(3), rtol, atol)
        np.testing.assert_allclose( 2, traj.getVal(5), rtol, atol)
        np.testing.assert_allclose( 3, traj.getVal(6), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(8), rtol, atol)
    def test_PeriodicCubicSpline_position_intermediary(self):
        traj = self.getDefaultPeriodicCubicSpline()
        np.testing.assert_allclose( 3.016, traj.getVal(0.5), rtol, atol)
        np.testing.assert_allclose(-0.141, traj.getVal(2.0), rtol, atol)
        np.testing.assert_allclose( 3.016, traj.getVal(5.5), rtol, atol)
        np.testing.assert_allclose(-0.141, traj.getVal(7.0), rtol, atol)
    def test_PeriodicCubicSpline_vel(self):
        traj = self.getDefaultPeriodicCubicSpline()
        np.testing.assert_allclose( 2.063, traj.getVal(0.25, 1), rtol, atol)
        np.testing.assert_allclose(-3.105, traj.getVal(2.50, 1), rtol, atol)
        np.testing.assert_allclose( 2.063, traj.getVal(5.25, 1), rtol, atol)
        np.testing.assert_allclose(-3.105, traj.getVal(7.50, 1), rtol, atol)
    def test_PeriodicCubicSpline_acc(self):
        traj = self.getDefaultPeriodicCubicSpline()
        np.testing.assert_allclose(-4.125, traj.getVal( 0.5, 2), rtol, atol)
        np.testing.assert_allclose( 4.416, traj.getVal( 2.7, 2), rtol, atol)
        np.testing.assert_allclose(-4.125, traj.getVal(10.5, 2), rtol, atol)
        np.testing.assert_allclose( 4.416, traj.getVal(12.7, 2), rtol, atol)
    def test_PeriodicCubicSpline_invalid_badLimits(self):
        with np.testing.assert_raises(Exception):
            self.getBadLimitsPeriodicCubicSpline()
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
        dicom_eval = TrajectoriesAssignment(args.path, g)
        root = dicom_eval.run(original_evaluation)
        root.exportToJson(json_path)
        txt_content = sae.evalToString(root)
        txt_path = join(args.path, g.getKey() + ".txt")
        with open(txt_path, "w") as f:
            f.write(txt_content)
        print(txt_content)

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

# With a constant spacing of 2 and cubic zero derivative, we can easily obtain
# values at middle of two nodes
# - pos(1) = (pos(2) + pos(0))
# - vel(1) = 3a+2b = b = 3*(x_end-s_src)/4
# - acc(1) = 6a+2b = 0
# - acc(0) = 2b = 6*(x_end-x_src) / 4
#
# Solving systems can be done easily manually:
#   - d = x_src
#   - c = 0
#   - 8a + 4b + d = x_end   -> -4a = (x_end - d) -> a = (d-xend) / 4
#   - 12a + 4b = 0  -> b = -3a  -> b = 3*(x_end - d) /4
rrr_targets = np.array([[0,0.8,0.0,1.05],[2,0.0,0.8,1.05],[4,0.0,0.8,0.65]])
dt = 1e-3

def getJointAcc(traj, t):
    # Initial version of trajectory did not contain JointAcceleration
    joint_acc = np.zeros(3)
    for i in range(3):
        joint_acc[i] = traj.getVal(t, i, 2, "joint")
    return joint_acc

def getOpAcc(traj, t):
    # Initial version of trajectory did not contain OperationalAcceleration
    op_acc = np.zeros(3)
    for i in range(3):
        op_acc[i] = traj.getVal(t, i, 2, "operational")
    return op_acc

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
        self.robot_traj_builder = getattr(traj_module, "RobotTrajectory")
        self.rrr_model = getattr(ctrl_module, "RRRRobot")()
        print(self.traj_builder)
        for c in self.root.children[1:3]:
            print("Evaluating: " + c.name)
            c.eval()

    def getStructure(self):
        return sae.EvalNode("3-Trajectoires", children=[
            self.getDefaultRulesTree(),
            self.get1DTrajectoriesTree(),
            self.getRobotTrajectoriesTree(),
            sae.EvalNode("Compte rendu", eval_func=sae.manualEval, max_points=3)
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
        node = sae.EvalNode("Trajectoires 1D",
                            children = [self.getTrajectoryTree(traj_name)
                                        for traj_name in trajectories])
        return node

    def getTrajectoryTree(self, traj_name):
        traj_node = sae.EvalNode(traj_name)
        dic = {
            "Degré" : "degree",
            "Constructeur invalide" : "invalid",
            "Limites" : "limits",
            "Durée" : "duration",
            "Position" : "position",
            "Vitesse" : "vel",
            "Accélération" : "acc"
        }
        coeff = 1
        if traj_name == "TrapezoidalVelocity":
            coeff *= 3
        for test_name, method_name in dic.items():
            pattern = "test_{:}_{:}".format(traj_name, method_name)
            methods = [m for m in dir(self) if pattern in m]
            if (len(methods) == 0):
                continue
            test_points = 0.1 * coeff
            if test_name in ["Position", "Vitesse", "Accélération"] :
                test_points = 0.2 * coeff
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

    def getRobotTrajectoriesTree(self):
        trajectories = [
            "RRRCubic0DJoint",
            "RRRCubic0DOperational"
        ]
        node = sae.EvalNode("Trajectoires pour robots",
                            children = [self.getRobotTrajectoryTree(traj_name)
                                        for traj_name in trajectories])
        return node

    def getRobotTrajectoryTree(self, traj_name):
        traj_node = sae.EvalNode(traj_name)
        dic = {
            "Position articulaire" : "jointPos",
            "Vitesse articulaire" : "jointVel",
            "Accélération articulaire" : "jointAcc",
            "Position opérationnelle" : "opPos",
            "Vitesse opérationnelle" : "opVel",
            "Accélération opérationnelle" : "opAcc"
        }
        for test_name, method_name in dic.items():
            pattern = "test_{:}_{:}".format(traj_name, method_name)
            methods = [m for m in dir(self) if pattern in m]
            test_points = 0.75
            if len(methods) == 0:
                continue
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
        np.testing.assert_allclose( 1, traj.getVal(0.33, 1), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(1.53, 1), rtol, atol)
    def test_LinearSpline_acc(self):
        traj = self.getDefaultLinearSpline()
        np.testing.assert_allclose(0, traj.getVal(0.33, 2), rtol, atol)
        np.testing.assert_allclose(0, traj.getVal(1.53, 2), rtol, atol)
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
        np.testing.assert_equal( 2, traj.getVal(0,0))
        np.testing.assert_equal( 3, traj.getVal(1,0))
        np.testing.assert_equal(-3, traj.getVal(3,0))
    def test_CubicZeroDerivativeSpline_position_intermediary(self):
        traj = self.getDefaultCubicZeroDerivativeSpline()
        np.testing.assert_allclose( 2.5, traj.getVal(0.5,0), rtol, atol)
        np.testing.assert_allclose( 0.0, traj.getVal(2.0,0), rtol, atol)
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
        np.testing.assert_allclose( 2, traj.getVal(0,0), rtol, atol)
        np.testing.assert_allclose( 3, traj.getVal(1,0), rtol, atol)
        np.testing.assert_allclose( 1, traj.getVal(2,0), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(3,0), rtol, atol)
        np.testing.assert_allclose(-2, traj.getVal(5,0), rtol, atol)
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
        np.testing.assert_equal( 2, traj.getVal(0,0))
        np.testing.assert_equal( 3, traj.getVal(1,0))
        np.testing.assert_equal(-3, traj.getVal(3,0))
    def test_CubicCustomDerivativeSpline_position_intermediary(self):
        traj = self.getDefaultCubicCustomDerivativeSpline()
        np.testing.assert_allclose( 2.75, traj.getVal(0.5,0), rtol, atol)
        np.testing.assert_allclose(-0.25, traj.getVal(2.0,0), rtol, atol)
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
        np.testing.assert_allclose( 2, traj.getVal(0,0), rtol, atol)
        np.testing.assert_allclose( 3, traj.getVal(1,0), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(3,0), rtol, atol)
    def test_NaturalCubicSpline_position_intermediary(self):
        traj = self.getDefaultNaturalCubicSpline()
        np.testing.assert_allclose(2.75, traj.getVal(0.5,0), rtol, atol)
        np.testing.assert_allclose(1.00, traj.getVal(2.0,0), rtol, atol)
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
        np.testing.assert_allclose( 2, traj.getVal(0,0), rtol, atol)
        np.testing.assert_allclose( 3, traj.getVal(1,0), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(3,0), rtol, atol)
        np.testing.assert_allclose( 2, traj.getVal(5,0), rtol, atol)
        np.testing.assert_allclose( 3, traj.getVal(6,0), rtol, atol)
        np.testing.assert_allclose(-3, traj.getVal(8,0), rtol, atol)
    def test_PeriodicCubicSpline_position_intermediary(self):
        traj = self.getDefaultPeriodicCubicSpline()
        np.testing.assert_allclose( 3.016, traj.getVal(0.5,0), rtol, atol)
        np.testing.assert_allclose(-0.141, traj.getVal(2.0,0), rtol, atol)
        np.testing.assert_allclose( 3.016, traj.getVal(5.5,0), rtol, atol)
        np.testing.assert_allclose(-0.141, traj.getVal(7.0,0), rtol, atol)
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
    def get2PointsUpTrapezoidalVelocity(self):
        # accTime: 0.2
        # accDist: 0.1
        # steadyTime: 0.8
        return self.traj_builder("TrapezoidalVelocity", 0.0,
                                 np.array([[2],[3]]),
                                 {"acc_max" : 5.0, "vel_max" : 1.0})
    def get2PointsDownTrapezoidalVelocity(self):
        # accTime: 0.2
        # accDist: -0.1
        # steadyTime: 1.8
        return self.traj_builder("TrapezoidalVelocity", 0.0,
                                 np.array([[3],[1]]),
                                 {"acc_max" : 5.0, "vel_max" : 1.0})
    def getNoSteadySpeedTrapezoidalVelocity(self):
        # accTime: 0.5
        # accDist: 0.5
        # steadyTime: 0.0
        return self.traj_builder("TrapezoidalVelocity", 0.0,
                                 np.array([[1],[2]]),
                                 {"acc_max" : 1.0, "vel_max" : 2.0})
    def get4PointsTrapezoidalVelocity(self):
        return self.traj_builder("TrapezoidalVelocity", 0.0,
                                 np.array([[2],[3],[1],[2]]),
                                 {"acc_max" : 5.0, "vel_max" : 1.0})
    def getBadKnotsTrapezoidalVelocity(self):
        return self.traj_builder("TrapezoidalVelocity", 0.0,
                                 np.array([[0,2],[3,3],[2,-2]]),
                                 {"acc_max" : 2.0, "vel_max" : 1.0})
    def getMissingParametersTrapezoidalVelocity(self):
        return self.traj_builder("TrapezoidalVelocity", 0.0,
                                 np.array([[0,2],[3,3],[2,-2]]))
    def test_TrapezoidalVelocity_duration_2pointsUp(self):
        traj = self.get2PointsUpTrapezoidalVelocity()
        np.testing.assert_allclose(1.2, traj.getEnd(), rtol, atol)
    def test_TrapezoidalVelocity_duration_2pointsDown(self):
        traj = self.get2PointsDownTrapezoidalVelocity()
        np.testing.assert_allclose(2.2, traj.getEnd(), rtol, atol)
    def test_TrapezoidalVelocity_duration_NoSteadySpeed(self):
        traj = self.getNoSteadySpeedTrapezoidalVelocity()
        np.testing.assert_allclose(2.0, traj.getEnd(), rtol, atol)
    def test_TrapezoidalVelocity_duration_4points(self):
        traj = self.get4PointsTrapezoidalVelocity()
        np.testing.assert_allclose(4.6, traj.getEnd(), rtol, atol)
    def test_TrapezoidalVelocity_position_2pointsUp(self):
        traj = self.get2PointsUpTrapezoidalVelocity()
        np.testing.assert_allclose(2.1, traj.getVal(0.2,0), rtol, atol)
        np.testing.assert_allclose(2.5, traj.getVal(0.6,0), rtol, atol)
        np.testing.assert_allclose(2.9, traj.getVal(1.0,0), rtol, atol)
    def test_TrapezoidalVelocity_position_2pointsDown(self):
        traj = self.get2PointsDownTrapezoidalVelocity()
        np.testing.assert_allclose(2.9, traj.getVal(0.2,0), rtol, atol)
        np.testing.assert_allclose(2.5, traj.getVal(0.6,0), rtol, atol)
        np.testing.assert_allclose(1.1, traj.getVal(2.0,0), rtol, atol)
    def test_TrapezoidalVelocity_position_NoSteadySpeed(self):
        traj = self.getNoSteadySpeedTrapezoidalVelocity()
        np.testing.assert_allclose(1.005, traj.getVal(0.1,0), rtol, atol)
        np.testing.assert_allclose(1.50, traj.getVal(1.0,0), rtol, atol)
        np.testing.assert_allclose(1.995, traj.getVal(1.9,0), rtol, atol)
    def test_TrapezoidalVelocity_position_4points(self):
        traj = self.get4PointsTrapezoidalVelocity()
        np.testing.assert_allclose(2.1, traj.getVal(0.2,0), rtol, atol)
        np.testing.assert_allclose(2.9, traj.getVal(1.4,0), rtol, atol)
        np.testing.assert_allclose(1.1, traj.getVal(3.6,0), rtol, atol)
    def test_TrapezoidalVelocity_vel_2pointsUp(self):
        traj = self.get2PointsUpTrapezoidalVelocity()
        np.testing.assert_allclose(0.5, traj.getVal(0.1,1), rtol, atol)
        np.testing.assert_allclose(1.0, traj.getVal(0.2,1), rtol, atol)
        np.testing.assert_allclose(1.0, traj.getVal(0.6,1), rtol, atol)
        np.testing.assert_allclose(0.5, traj.getVal(1.1,1), rtol, atol)
    def test_TrapezoidalVelocity_vel_2pointsDown(self):
        traj = self.get2PointsDownTrapezoidalVelocity()
        np.testing.assert_allclose(-0.5, traj.getVal(0.1,1), rtol, atol)
        np.testing.assert_allclose(-1.0, traj.getVal(0.2,1), rtol, atol)
        np.testing.assert_allclose(-1.0, traj.getVal(0.6,1), rtol, atol)
        np.testing.assert_allclose(-0.5, traj.getVal(2.1,1), rtol, atol)
    def test_TrapezoidalVelocity_vel_NoSteadySpeed(self):
        traj = self.getNoSteadySpeedTrapezoidalVelocity()
        np.testing.assert_allclose(0.2, traj.getVal(0.2,1), rtol, atol)
        np.testing.assert_allclose(1.0, traj.getVal(1.0,1), rtol, atol)
        np.testing.assert_allclose(0.3, traj.getVal(1.7,1), rtol, atol)
    def test_TrapezoidalVelocity_vel_4points(self):
        traj = self.get4PointsTrapezoidalVelocity()
        np.testing.assert_allclose( 0.5, traj.getVal(0.1,1), rtol, atol)
        np.testing.assert_allclose(-0.5, traj.getVal(1.3,1), rtol, atol)
        np.testing.assert_allclose(-1.0, traj.getVal(1.6,1), rtol, atol)
        np.testing.assert_allclose( 0.5, traj.getVal(4.5,1), rtol, atol)
    def test_TrapezoidalVelocity_acc_2pointsUp(self):
        traj = self.get2PointsUpTrapezoidalVelocity()
        np.testing.assert_allclose( 5.0, traj.getVal(0.1,2), rtol, atol)
        np.testing.assert_allclose( 0.0, traj.getVal(0.6,2), rtol, atol)
        np.testing.assert_allclose(-5.0, traj.getVal(1.1,2), rtol, atol)
    def test_TrapezoidalVelocity_acc_2pointsDown(self):
        traj = self.get2PointsDownTrapezoidalVelocity()
        np.testing.assert_allclose(-5.0, traj.getVal(0.1,2), rtol, atol)
        np.testing.assert_allclose( 0.0, traj.getVal(0.3,2), rtol, atol)
        np.testing.assert_allclose( 5.0, traj.getVal(2.1,2), rtol, atol)
    def test_TrapezoidalVelocity_acc_NoSteadySpeed(self):
        traj = self.getNoSteadySpeedTrapezoidalVelocity()
        np.testing.assert_allclose( 1.0, traj.getVal(0.2,2), rtol, atol)
        np.testing.assert_allclose(-1.0, traj.getVal(1.1,2), rtol, atol)
    def test_TrapezoidalVelocity_acc_4points(self):
        traj = self.get4PointsTrapezoidalVelocity()
        np.testing.assert_allclose( 5.0, traj.getVal(0.1,2), rtol, atol)
        np.testing.assert_allclose(-5.0, traj.getVal(1.3,2), rtol, atol)
        np.testing.assert_allclose( 5.0, traj.getVal(3.3,2), rtol, atol)
        np.testing.assert_allclose( 0.0, traj.getVal(4.2,2), rtol, atol)
    def test_TrapezoidalVelocity_invalid_badKnots(self):
        with np.testing.assert_raises(Exception):
            self.getBadKnotsTrapezoidalVelocity()
    def test_TrapezoidalVelocity_invalid_missingParameters(self):
        with np.testing.assert_raises(Exception):
            self.getMissingParametersTrapezoidalVelocity()
    def getRRRCubic0DJointTrajectory(self):
        return self.robot_traj_builder(self.rrr_model, rrr_targets.copy(),
                                       "CubicZeroDerivativeSpline",
                                       "operational", "joint")
    def test_RRRCubic0DJoint_jointPos(self):
        traj = self.getRRRCubic0DJointTrajectory()
        for idx in range(rrr_targets.shape[0]):
            t = rrr_targets[idx,0]
            exp_op = rrr_targets[idx,1:]
            joint_pos = traj.getJointTarget(t)
            op_pos = self.rrr_model.computeMGD(joint_pos)
            np.testing.assert_allclose(exp_op, op_pos, rtol, atol)
        # Testing that intermediary pos is in middle of prev and next
        # (planification in joint space)
        for idx in range(rrr_targets.shape[0]-1):
            prevT = rrr_targets[idx,0]
            nextT = rrr_targets[idx+1,0]
            prevJoint = traj.getJointTarget(prevT)
            nextJoint = traj.getJointTarget(nextT)
            exp_pos = (nextJoint+prevJoint) / 2
            joint_pos = traj.getJointTarget((prevT+nextT)/2)
            np.testing.assert_allclose(exp_pos, joint_pos, rtol, atol)
    def test_RRRCubic0DJoint_opPos(self):
        traj = self.getRRRCubic0DJointTrajectory()
        for idx in range(rrr_targets.shape[0]):
            t = rrr_targets[idx,0]
            exp_op = rrr_targets[idx,1:]
            op_pos = traj.getOperationalTarget(t)
            np.testing.assert_allclose(exp_op, op_pos, rtol, atol)
        # Testing that intermediary pos is in middle of prev and next
        # (planification in joint space)
        for idx in range(rrr_targets.shape[0]-1):
            prevT = rrr_targets[idx,0]
            nextT = rrr_targets[idx+1,0]
            prevJoint = traj.getJointTarget(prevT)
            nextJoint = traj.getJointTarget(nextT)
            exp_pos = self.rrr_model.computeMGD((nextJoint+prevJoint) / 2)
            joint_pos = traj.getOperationalTarget((prevT+nextT)/2)
            np.testing.assert_allclose(exp_pos, joint_pos, rtol, atol)
    def test_RRRCubic0DJoint_jointVel(self):
        traj = self.getRRRCubic0DJointTrajectory()
        # Checking that on all control points, speed is 0
        for idx in range(rrr_targets.shape[0]):
            t = rrr_targets[idx,0]
            exp_vel = np.zeros(3)
            vel = traj.getJointVelocity(t)
            np.testing.assert_allclose(exp_vel, vel, rtol, atol)
        # Checking values at the middle of each target
        for idx in range(rrr_targets.shape[0]-1):
            prevT = rrr_targets[idx,0]
            nextT = rrr_targets[idx+1,0]
            prevJoint = traj.getJointTarget(prevT)
            nextJoint = traj.getJointTarget(nextT)
            t = (rrr_targets[idx+1,0] + rrr_targets[idx,0]) / 2
            exp_vel = 3*(nextJoint - prevJoint)/4
            vel = traj.getJointVelocity(t)
            np.testing.assert_allclose(exp_vel, vel, rtol, atol)
        # Checking consistency
        start = 1.8
        end = 2.2
        joint_pos = traj.getJointTarget(start)
        for t in np.arange(start,end,dt):
            exp_joint_pos = traj.getJointTarget(t)
            np.testing.assert_allclose(exp_joint_pos, joint_pos, rtol, atol)
            joint_pos += dt * traj.getJointVelocity(t)
    def test_RRRCubic0DJoint_opVel(self):
        traj = self.getRRRCubic0DJointTrajectory()
        # Checking that on all control points, speed is 0
        for idx in range(rrr_targets.shape[0]):
            t = rrr_targets[idx,0]
            exp_vel = np.zeros(3)
            vel = traj.getOperationalVelocity(t)
            np.testing.assert_allclose(exp_vel, vel, rtol, atol)
        # Checking consistency
        start = 1.8
        end = 2.2
        op_pos = traj.getOperationalTarget(start)
        for t in np.arange(start,end,dt):
            exp_op_pos = traj.getOperationalTarget(t)
            np.testing.assert_allclose(exp_op_pos, op_pos, rtol, atol)
            op_pos += dt * traj.getOperationalVelocity(t)
    def test_RRRCubic0DJoint_jointAcc(self):
        traj = self.getRRRCubic0DJointTrajectory()
        start = 1.8
        end = 2.2
        joint_vel = traj.getJointVelocity(start)
        for t in np.arange(start,end,dt):
            exp_joint_vel = traj.getJointVelocity(t)
            np.testing.assert_allclose(exp_joint_vel, joint_vel, rtol, atol)
            joint_vel += dt * getJointAcc(traj, t)
    def test_RRRCubic0DJoint_opAcc(self):
        traj = self.getRRRCubic0DJointTrajectory()
        # Checking that at start, acceleration is what we expect
        # v ~= 0 -> simplification, \dot{J} \dot{q} disappear
        start_t = 1e-8#Taking minimum value to avoid risk of confusion with<=
        q = traj.getJointTarget(start_t)
        q_end = traj.getJointTarget(2.0)
        exp_q_acc = 6 * (q_end-q)/4
        exp_J = self.rrr_model.computeJacobian(q)
        exp_op_acc = exp_J @ exp_q_acc
        op_acc = getOpAcc(traj,start_t)
        np.testing.assert_allclose(exp_op_acc, op_acc, rtol, atol)
        # Checking consistency
        start = 1.8
        end = 2.2
        op_vel = traj.getOperationalVelocity(start)
        for t in np.arange(start,end,dt):
            exp_op_vel = traj.getOperationalVelocity(t)
            np.testing.assert_allclose(exp_op_vel, op_vel, rtol, atol)
            op_vel += dt * getOpAcc(traj, t)
    def getRRRCubic0DOperationalTrajectory(self):
        return self.robot_traj_builder(self.rrr_model, rrr_targets.copy(),
                                       "CubicZeroDerivativeSpline",
                                       "operational", "operational")
    def test_RRRCubic0DOperational_jointPos(self):
        traj = self.getRRRCubic0DOperationalTrajectory()
        for idx in range(rrr_targets.shape[0]):
            t = rrr_targets[idx,0]
            exp_op = rrr_targets[idx,1:]
            joint_pos = traj.getJointTarget(t)
            op_pos = self.rrr_model.computeMGD(joint_pos)
            np.testing.assert_allclose(exp_op, op_pos, rtol, atol)
    def test_RRRCubic0DOperational_opPos(self):
        traj = self.getRRRCubic0DOperationalTrajectory()
        for idx in range(rrr_targets.shape[0]):
            t = rrr_targets[idx,0]
            exp_op = rrr_targets[idx,1:]
            op_pos = traj.getOperationalTarget(t)
            np.testing.assert_allclose(exp_op, op_pos, rtol, atol)
        # Testing that intermediary pos is in middle of op space
        for idx in range(rrr_targets.shape[0]-1):
            t = (rrr_targets[idx,0] + rrr_targets[idx+1,0]) / 2
            exp_op = (rrr_targets[idx,1:] + rrr_targets[idx+1,1:])/2
            op_pos = traj.getOperationalTarget(t)
            np.testing.assert_allclose(exp_op, op_pos, rtol, atol)
    def test_RRRCubic0DOperational_jointVel(self):
        traj = self.getRRRCubic0DOperationalTrajectory()
        # Checking that on all control points, speed is 0
        for idx in range(rrr_targets.shape[0]):
            t = rrr_targets[idx,0]
            exp_vel = np.zeros(3)
            vel = traj.getJointVelocity(t)
            np.testing.assert_allclose(exp_vel, vel, rtol, atol)
        # Checking consistency
        start = 1.5
        end = 2.5
        joint_pos = traj.getJointTarget(start)
        for t in np.arange(start,end,dt):
            exp_joint_pos = traj.getJointTarget(t)
            np.testing.assert_allclose(exp_joint_pos, joint_pos, rtol, atol)
            joint_pos += dt * traj.getJointVelocity(t)
    def test_RRRCubic0DOperational_opVel(self):
        traj = self.getRRRCubic0DOperationalTrajectory()
        traj = self.getRRRCubic0DOperationalTrajectory()
        # Checking that on all control points, speed is 0
        for idx in range(rrr_targets.shape[0]):
            t = rrr_targets[idx,0]
            exp_vel = np.zeros(3)
            vel = traj.getOperationalVelocity(t)
            np.testing.assert_allclose(exp_vel, vel, rtol, atol)
        # Checking values at the middle of each target
        for idx in range(rrr_targets.shape[0]-1):
            t = (rrr_targets[idx+1,0] + rrr_targets[idx,0]) / 2
            exp_vel = 3*(rrr_targets[idx+1,1:]- rrr_targets[idx,1:])/4
            vel = traj.getOperationalVelocity(t)
            np.testing.assert_allclose(exp_vel, vel, rtol, atol)
        # Checking consistency
        start = 1.8
        end = 2.2
        op_pos = traj.getOperationalTarget(start)
        for t in np.arange(start,end,dt):
            exp_op_pos = traj.getOperationalTarget(t)
            np.testing.assert_allclose(exp_op_pos, op_pos, rtol, atol)
            op_pos += dt * traj.getOperationalVelocity(t)
    def test_RRRCubic0DOperational_jointAcc(self):
        traj = self.getRRRCubic0DOperationalTrajectory()
        start = 1.8
        end = 2.2
        joint_vel = traj.getJointVelocity(start)
        for t in np.arange(start,end,dt):
            exp_joint_vel = traj.getJointVelocity(t)
            np.testing.assert_allclose(exp_joint_vel, joint_vel, rtol, atol)
            joint_vel += dt * getJointAcc(traj, t)
    def test_RRRCubic0DOperational_opAcc(self):
        traj = self.getRRRCubic0DOperationalTrajectory()
        start = 1.8
        end = 2.2
        op_vel = traj.getOperationalVelocity(start)
        for t in np.arange(start,end,dt):
            exp_op_vel = traj.getOperationalVelocity(t)
            np.testing.assert_allclose(exp_op_vel, op_vel, rtol, atol)
            op_vel += dt * getOpAcc(traj, t)

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

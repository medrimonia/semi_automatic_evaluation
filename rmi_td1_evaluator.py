#!/usr/bin/env python3

import argparse
import os
import sys
from os.path import join, dirname
from anytree.importer import JsonImporter

import semi_automatic_evaluator as sae
import numpy as np
from scipy.spatial.transform import Rotation as R

from importlib import reload, import_module

rtol = 1e-6
atol = 1e-6

ht_module = None
ctrl_module = None

class MGDAssignment(sae.EvaluationProcess):
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
        self.rot_x = getattr(ht_module, "rot_x")
        self.rot_y = getattr(ht_module, "rot_y")
        self.rot_z = getattr(ht_module, "rot_z")
        self.translation = getattr(ht_module, "translation")
        self.invert_transform = getattr(ht_module, "invert_transform")
        self.get_quat = getattr(ht_module, "get_quat")
        self.rt_robot = getattr(ctrl_module, "RTRobot")
        self.rrr_robot = getattr(ctrl_module, "RRRRobot")
        for c in self.root.children[1:]:
            print("Evaluating: " + c.name)
            c.eval()

    def getStructure(self):
        return sae.EvalNode("1-MGD Assignment", children=[
            self.getDefaultRulesTree(),
            self.getHTTree(),
            self.getRobotTree("RT"),
            self.getRobotTree("RRR"),
            self.getToolsTree(),
            sae.EvalNode("Compte rendu", eval_func=sae.manualEval, max_points=2)
            ])

    def getHTTree(self):
        return sae.EvalNode("Homogeneous Transform", children=[
            self.getRotXTree(),
            self.getRotYTree(),
            self.getRotZTree(),
            self.getTranslationTree(),
            self.getInvertTransformTree(),
            self.getGetQuatTree()
        ])

    def getRobotTree(self, robot_name):
        return sae.EvalNode("Robot {:}".format(robot_name),
                            set_up_func= lambda : print(
                                "From {:}, run './simulation --robot {:}'".format(
                                    self.group_path, robot_name)
                            ),
                            children=[
                                sae.EvalNode("Tool position",
                                             eval_func=sae.manualEval),
                                sae.EvalNode("Tool orientation",
                                             eval_func=sae.manualEval),
                                sae.EvalNode("MGD",
                                             eval_func=lambda node : sae.assertionEval(node,
                                                                                       lambda :
                                                                                       self.test_mgd(robot_name)))
                            ])

    def getToolsTree(self):
        return sae.EvalNode("Tools", children=[
            sae.EvalNode("DebugLine", eval_func=sae.manualEval,
                         set_up_func=lambda : print("If memory is not limited, half points.")),
            sae.EvalNode("Log joints measured", eval_func=sae.manualEval),
            sae.EvalNode("Log operational pos", eval_func=sae.manualEval)
            ])

    def getRotXTree(self):
        return sae.EvalNode("rot_x", children=[
            sae.EvalNode("rot_x(0)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_rot_X_0)),
            sae.EvalNode("rot_x(pi/2)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_rot_X_halfpi))
        ])

    def getRotYTree(self):
        return sae.EvalNode("rot_y", children=[
            sae.EvalNode("rot_y(0)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_rot_Y_0)),
            sae.EvalNode("rot_y(pi/2)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_rot_Y_halfpi))
        ])

    def getRotZTree(self):
        return sae.EvalNode("rot_z", children=[
            sae.EvalNode("rot_z(0)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_rot_Z_0)),
            sae.EvalNode("rot_z(pi/2)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_rot_Z_halfpi))
        ])

    def getTranslationTree(self):
        return sae.EvalNode("translation", children=[
            sae.EvalNode("translation(0)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_translation_0)),
            sae.EvalNode("translation(pi/2)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_translation_other))
        ])

    def getInvertTransformTree(self):
        return sae.EvalNode("invert_transform", children=[
            sae.EvalNode("invert_transform(I)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_invert_transform_eye)),
            sae.EvalNode("invert_transform(rot_x(pi/3))",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_invert_transform_rot)),
            sae.EvalNode("invert_transform(translation([1,0,-2]))",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_invert_transform_translation)),
            sae.EvalNode("invert_transform(...)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_invert_transform_mixed))
        ])

    def getGetQuatTree(self):
        return sae.EvalNode("get_quat", children=[
            sae.EvalNode("get_quat(I)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_get_quat_eye)),
            sae.EvalNode("get_quat(...)",
                         max_points=0.5,
                         eval_func= lambda node : sae.assertionEval(node,self.test_get_quat_rot_other))
        ])

    def test_rot_X_0(self):
        received = self.rot_x(0)
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rot_X_halfpi(self):
        received = self.rot_x(np.pi/2)
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, -1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rot_Y_0(self):
        received = self.rot_y(0)
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rot_Y_halfpi(self):
        received = self.rot_y(np.pi/2)
        expected = np.array([[0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [-1, 0, 0, 0],
                             [0, 0, 0, 1]], dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rot_Z_0(self):
        received = self.rot_z(0)
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_rot_Z_halfpi(self):
        received = self.rot_z(np.pi/2)
        expected = np.array([[0, -1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_translation_0(self):
        received = self.translation(np.array([0,0,0]))
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_translation_other(self):
        vec = np.array([1,-2,0])
        received = self.translation(vec)
        expected = np.eye(4, dtype=np.double)
        for i in range(3):
            expected[i,3] = vec[i]
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_invert_transform_eye(self):
        arg = np.eye(4, dtype=np.double)
        received = self.invert_transform(arg)
        expected = np.eye(4, dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_invert_transform_rot(self):
        alpha = np.pi/3
        T = rot_x(alpha)
        received = self.invert_transform(T)
        expected = rot_x(-alpha)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_invert_transform_translation(self):
        vec = np.array([1,0,-2])
        T = translation(vec)
        received = self.invert_transform(T)
        expected = translation(-vec)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_invert_transform_mixed(self):
        a1 = np.pi/3
        v1 = np.array([1,0,-2])
        a2 = -np.pi/4
        v2 = np.array([-1,2,-5])
        T = rot_x(a1) * translation(v1) * rot_y(a2) * translation(v2)
        received = self.invert_transform(T)
        expected = translation(-v2) * rot_y(-a2) * translation(-v1) * rot_x(-a1)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_get_quat_eye(self):
        received = get_quat(np.eye(4, dtype=np.double))
        expected = np.array([0,0,0,1], dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_get_quat_eye(self):
        received = get_quat(np.eye(4, dtype=np.double))
        expected = np.array([0,0,0,1], dtype=np.double)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_get_quat_rot_other(self):
        T = rot_x(np.pi/3) @ rot_y(-np.pi/3) @ rot_z(np.pi/2)
        received = self.get_quat(T)
        expected = get_quat(T)
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_mgd(self, robot_name):
        if robot_name == "RRR":
            self.test_mgd_rrr()
        elif robot_name == "RT":
            self.test_mgd_rt()

    def test_mgd_rt(self):
        robot = self.rt_robot()
        joints = [np.pi/2,0.1]
        print("Testing rt with joints:")
        print(joints)
        received = robot.computeMGD(joints)
        expected = np.array([0.25,0.3])
        np.testing.assert_allclose(received, expected, rtol, atol)

    def test_mgd_rrr(self):
        robot = self.rrr_robot()
        joints = np.array([np.pi/2,np.pi/2,-np.pi/2])
        print("Testing rrr with joints:")
        print(joints)
        received = robot.computeMGD(joints)
        expected = np.array([-0.71,0.0,1.31])
        np.testing.assert_allclose(received, expected, rtol, atol)

# IMPLEMENTING DEFAULT VERSION TO TEST BEHAVIORS
# ----------------------------------------------

def rot_x(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around x
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]], dtype=np.double)

def rot_y(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around y
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=np.double)

def rot_z(alpha):
    """Return the 4x4 homogeneous transform corresponding to a rotation of
    alpha around z
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.double)

def translation(vec):
    """Return the 4x4 homogeneous transform corresponding to a translation of
    vec
    """
    return np.array([[1, 0, 0, vec[0]],
                     [0, 1, 0, vec[1]],
                     [0, 0, 1, vec[2]],
                     [0, 0, 0, 1]], dtype=np.double)

def get_quat(T):
    """
    Parameters
    ----------
    T : np.ndarray shape(4,4)
        A 3d homogeneous transformation matrix

    Returns
    -------
    quat : np.ndarray shape(4,)
        a quaternion representing the rotation part of the homogeneous
        transformation matrix
    """
    return R.from_dcm(T[:3,:3]).as_quat()

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
        dicom_eval = MGDAssignment(args.path, g)
        root = dicom_eval.run(original_evaluation)
        root.exportToJson(json_path)
        txt_content = sae.evalToString(root)
        txt_path = join(args.path, g.getKey() + ".txt")
        with open(txt_path, "w") as f:
            f.write(txt_content)
        print(txt_content)

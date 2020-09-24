#!/usr/bin/env python3

from anytree import NodeMixin, RenderTree
import json
import io
import os
import tarfile
import subprocess
import sys
import traceback
import abc

from os.path import isfile, join

from group_work_packer import *
from terminal_utils import *

"""
Documentation in progress...

Examples of generated outputs: TO BE DESCRIBED

Module classes:

- Student: The core properties of a student
- Group: A simple list of students
- GroupCollection: A list of student groups
- Eval: Simple evaluation on an exercise
- EvalNode: The core element of the evaluation is the node of a tree
  - Can be exported to both, result
- EvaluationProcess: This is the class that should be inherited from when creating an assignment
- Assignment: The results of all the group on a specific assignment.
  - Results can be exported as json files or csv files
- TeachingUnit: A class allowing to regroup multiples assignment and to produce
  a recap for all the students

Different ways to evaluate a task: TO BE DESCRIBED

Handling invalid configurations: TO BE DESCRIBED

Different languages: TO BE DESCRIBED
"""

SAE_LG="EG"

messages= {
    "NO_ARCHIVE" : {
        "EG" : "No archive file received",
        "FR" : "Pas d'archive reçue"
    },
    "FAILED_EXTRACTION": {
        "EG" : "Unable to extract archive automatically",
        "FR" : "Impossible d'extraire l'archive automatiquement"
    },
    "AND" : {
        "EG" : "AND",
        "FR" : "ET"
    },
    "ARCHIVE_FOLDER_FAILED": {
        "EG" : "All files should be in a single folder",
        "FR" : "Tous les fichiers doivent être dans un dossier unique"
    },
    "SEE": {
        "EG" : "See",
        "FR" : "Voir"
    }
}

def getMessage(id):
    if id not in messages:
        return "<UNKNOWN MESSAGE ID: {:}>".format(id)
    if SAE_LG not in messages[id]:
        return "<UNSUPPORTED LANGUAGE '{:}' for message id '{:}'>".format(SAE_LG, id)
    return messages[id][SAE_LG]

def tarOpenUTF8Proof(path):
    """
    Open a utf8 file and returns the associated tarfile.TarFile while
    supporting utf8 encoding in paths
    """
    # from: https://issue.life/questions/57723988
    with open(path, 'rb') as f:
        return tarfile.open(fileobj=io.BytesIO(f.read()),mode='r')


class Eval:
    """
    A simple class representing an exercise

    Members
    -------
    points: float
        The number of points achieved on the exercise
    max_points: float
        The maximal number of points that can be obtained on the exercise
    msg: str
        The name of the exercise
    evaluated: bool
        Has the node already been evaluated or has it just been created
    eval_func: lambda EvalNode : None
        An evaluation function which updates the node points and the node
        message
    """
    def __init__(self, max_points = 1.0, points = 0, msg="", evaluated = False):
        self.max_points = max_points
        self.points = points
        self.msg = msg
        self.evaluated = evaluated


class EvalNode(Eval, NodeMixin):
    def __init__(self, name, max_points=1.0,points = 0,  msg="",
                 eval_func = None, set_up_func = None, tear_down_func = None,
                 evaluated = False, parent=None, children=None):
        super().__init__()
        Eval.__init__(self, max_points, points, msg, evaluated)
        self.name = name
        self.parent = parent
        self.eval_func = eval_func
        self.set_up_func = set_up_func
        self.tear_down_func = tear_down_func
        if children is not None:
            self.children = children
        self.syncPoints()

    def syncPoints(self):
        if len(self.children) == 0:
            return
        self.points = 0
        self.max_points = 0
        for c in self.children:
            c.syncPoints();
            self.points += c.points
            self.max_points += c.max_points

    def setSuccess(self, successRatio=1.0):
        self.points = successRatio * self.max_points

    def setMessageRecursive(self, msg):
        self.msg = msg
        for node in self.descendants:
            node.msg = msg

    def checkRecursive(self):
        for node in self.descendants:
            if node.is_leaf:
                checkNode(node)

    def checkIfBuilt(self, build_success):
        if build_success:
            self.checkRecursive()
        else:
            self.setMessageRecursive("Failed to build")

    def eval(self):
        if self.set_up_func is not None:
            self.set_up_func()
        for c in self.children:
            c.eval()
        if self.eval_func is not None:
            self.eval_func(self)
        if self.tear_down_func is not None:
            self.tear_down_func()

class GroupCollection(list):
    def __init__(self):
        self = []

    @staticmethod
    def loadFromJson(self, path):
        """
        Parameters
        ----------
        path : str
            The path to a json file containing an array of groups

        Returns
        -------
        collection : GroupCollection
            The collection build from the json file
        """
        collection = GroupCollection()
        with open(path) as f:
            val = json.load(f)
            if not isinstance(val, list):
                raise RuntimeError("GroupCollection should be an array in file {:}".format(path))
            for group in val:
                if not isinstance(group, list):
                    raise RuntimeError("Each group should be an array in file {:}".format(path))
                g = Group()
                for student in group:
                    if not isinstance(student, list):
                        raise RuntimeError("Each student should be an array in file {:}".format(path))
                    if len(student) != 2:
                        raise RuntimeError("Expecting an element of size 2 in file {:}".format(path))
                    g.append(Student(student[0], student[1]))
                collection.append(g)

    @staticmethod
    def discover(path):
        """
        Analyze the content of a directory and consider that each `tar.gz` file
        in it is a different group. Extract all the archives and look up for the
        `group.csv` file in it to extract names of the group.

        Parameters
        ----------
        path : str
            The path to the directory

        Returns
        -------
        collection : GroupCollection
            The list of groups discovered
        """
        archives = [f for f in os.listdir(path) if isfile(join(path,f)) and f.endswith(".tar.gz")]
        archives.sort()
        group_collection = GroupCollection()
        for a in archives:
            group_name = a.replace(".tar.gz", "")
            try:
                with tarOpenUTF8Proof(join(path,a)) as tar:
                    group_inner_path = join(group_name, "group.csv")
                    with tar.extractfile(group_inner_path) as group_file:
                        if group_file is None:
                            print("Archive '" + a +
                                  "' failed to read file '" +
                                  group_inner_path + "'", file = sys.stderr)
                        else:
                            reader = csv.DictReader(io.StringIO(group_file.read().decode('utf-8')),
                                                    fieldnames=["LastName","FirstName"])
                            students = []
                            for row in reader:
                                students.append(Student(row["LastName"],row["FirstName"]))
                            group_collection.append(Group(students))
            except tarfile.ReadError as error:
                print("Failed to extract archive in '" + a + "': " + str(error))
        return group_collection

class EvaluationProcess:
    def __init__(self):
        self.__archive_path = None
        pass

    def run(self, path, group, original_eval = None):
        """
        Run a complete evaluation for the given group

        Parameters
        ----------
        group : Group
            The group which will be evaluated
        original_eval : EvalNode
            The root of a previous evaluation.
            TODO If provided, the evaluation is simply resumed
        """
        print("Running evaluation for group:")
        for s in group.students:
            print("->" + s.last_name + ", " + s.first_name)
        self.__archive_path = group.findArchive(path, ".tar.gz")
        self.root = self.getStructure()
        try:
            self._set_up()
        except Exception as exc:
            print ("Failed to set up environment with the following error:")
            print(exc)
            if not question("Would you like to proceed with evaluation?"):
                return self.root
        self._eval()
        self._tear_down()
        self.root.syncPoints()
        return self.root

    @abc.abstractmethod
    def getStructure(self):
        """
        Return the evaluation tree prior to any evaluation
        """

    def getDefaultRulesTree(self):
        """
        Return a tree for evaluation regarding respect of the rules
        """
        rules_root = EvalNode("Rules", children = [
            EvalNode("Mail title", 1.0, eval_func = manualEval),
            EvalNode("Mail recipients", 1.0, eval_func = manualEval),
            EvalNode("Archive name", 1.0, eval_func = manualEval,
                     set_up_func= lambda : print("Archive name: " + self.__archive_path)),
            EvalNode("Useful content", 1.0, eval_func = manualEval,
                     set_up_func= lambda : tarOpenUTF8Proof(self.__archive_path).list())
        ])
        return rules_root

    def _set_up(self):
        """
        Perform tasks which needs to be run prior to evaluation, default is
        extracting archive in its own folder
        """
        dst = os.path.dirname(self.__archive_path)
        with  tarOpenUTF8Proof(self.__archive_path) as tar:
            tar.extractall(dst)
        return None

    def _eval(self):
        """
        Run the evaluation process, default is to launch eval procedure on the
        whole tree
        """
        self.root.eval()

    def _tear_down(self):
        """
        Cleans the environment after an evaluation has been performed
        """
        return None

class Assignment:
    """
    Members
    -------
    groups : GroupCollection
        The list of the groups involved in this assignment
    evaluations : dict [str,EvalNode]
        A correspondance between group keys and their evaluation
    """

    def __init__(self):
        self.groups = GroupCollection()
        self.evaluations = {}
        #TODO shuffle for evaluation




def evalToString(eval_root):
    """
    Convert evaluation to a printable version
    """
    result_txt = ""
    max_width = 80
    max_tree_width = 0
    oversized_messages = []
    for pre, _, node in RenderTree(eval_root):
        line_width = len(u"%s%s" % (pre, node.name))
        if max_tree_width < line_width:
            max_tree_width = line_width
    max_msg_width = max_width - max_tree_width - 10 - 5
    for pre, _, node in RenderTree(eval_root):
        treestr = u"%s%s" % (pre, node.name)
        line = "{:} {:5.2f} / {:5.2f}".format(treestr.ljust(max_tree_width), node.points, node.max_points)
        if not node.msg is None:
            if len(node.msg) <= max_msg_width:
                line += " " + node.msg
            else:
                msg_index = len(oversized_messages) + 1
                line += " {:} *{:}".format(getMessage("SEE"),msg_index)
                oversized_messages.append(node.msg)
        result_txt += line + "\n"
    for idx in range(len(oversized_messages)):
        result_txt += "*{:}: {:}\n".format(idx+1, oversized_messages[idx])
    return result_txt


def manualEval(node):
    result = prompt("Is '{:}' valid? y(es), n(o), p(artially)".format(node.name), ['y','n','p'])
    if (result == 'y'):
        node.points = node.max_points
    else:
        if (result == 'p'):
            node.points = askFloat("How many points? (max_points={:})".format(node.max_points))
        node.msg = freeTextQuestion("What is the problem?")

def setRecursiveMessage(node, msg):
    for children in self.result.descendants:
        children.msg = msg

def systemCall(cmd):
    proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr= subprocess.PIPE, shell=True)
    stdout, stderr = proc.communicate()
    return [proc.returncode, stdout.decode("utf8").strip(), stderr.decode("utf8").strip()]

def approxEq(d1, d2, epsilon = 10**-6):
    return abs(d1 -d2) <= epsilon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to the directory")
    args = parser.parse_args()

    os.chdir(args.path)

    print(GroupCollection.discover("./"))

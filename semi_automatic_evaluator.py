from anytree import NodeMixin, RenderTree
import json
import io
import os
import tarfile
import subprocess
import sys
import traceback

from os.path import isfile, join

from group_work_packer import *

"""
Documentation in progress...

Examples of generated outputs: TO BE DESCRIBED

Module classes:

- Eval: Simple evaluation on an exercise
- EvalNode: The core element of the evaluation is the node of a tree
  - Can be exported to both, result
- GroupCollection: A list of student groups
- Assignment: The results of all the group on a specific task, this is the
  class that should be inherited from when creating an assignment
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
    """
    def __init__(self, points = 0, max_points = 1, msg=""):
        self.points = points
        self.max_points = max_points
        self.msg = msg

class EvalNode(Eval, NodeMixin):
    def __init__(self, name, points = 0, max_points=1, msg="", parent=None, children=None):
        super().__init__()
        Eval.__init__(self, points, max_points, msg)
        self.name = name
        self.parent = parent
        if children:
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
            # Handling utf-8 enconding in paths:
            # from: https://issue.life/questions/57723988
            with open(join(path,a), 'rb') as f:
                try:
                    tar = tarfile.open(fileobj=io.BytesIO(f.read()),mode='r')
                    group_inner_path = join(group_name, "group.csv")
                    with tar.extractfile(group_inner_path) as group_file:
                        if group_file is None:
                            print("Archive '" + a +
                                  "' failed to read file '" +
                                  group_inner_path + "'", file = sys.stderr)
                        else:
                            reader = csv.DictReader(io.StringIO(group_file.read().decode('utf-8')),
                                                    fieldnames=["LastName","FirstName"])
                            group = Group()
                            for row in reader:
                                group.students.append(Student(row["LastName"],row["FirstName"]))
                            group_collection.append(group)
                except tarfile.ReadError as error:
                    print("Failed to extract archive in '" + a + "': " + str(error))
        return group_collection

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

def prompt(msg, options):
    """
    Send a message to the user and wait for him to choose among the given options"""
    while True:
        print(msg)
        print(" ", end='')
        print("> ", end='')
        sys.stdout.flush()
        answer = sys.stdin.readline().strip().lower()
        if answer in options:
            return answer
        print("You should answer one of the following: {:}".format(options))

def question(text):
    """
    Ask a 'yes' or 'no' question to the user and returns true if answer was yes
    """
    return prompt(text, ['y','n']) == 'y'

def freeTextQuestion(msg):
    print(msg)
    print("> ", end='')
    sys.stdout.flush()
    return sys.stdin.readline().strip()

def askFloat(msg):
    print(msg)
    while True:
        print("> ", end='')
        sys.stdout.flush()
        try:
            return float(sys.stdin.readline().strip().lower())
        except ValueError as e:
            print("Not a valid number " + str(e))

def checkNode(node):
    result = prompt("Is: '{:}' ok? (y(es), n(o), p(artially)".format(node.name), ['y','n','p'])
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

    print(GroupCollection.discover(args.path))

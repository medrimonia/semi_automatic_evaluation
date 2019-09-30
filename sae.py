# SAE: Semi-Automatic Evaluation

from anytree import NodeMixin, RenderTree
import json
import os
import subprocess
import sys
import traceback

class Eval:
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

class Student:
    def __init__(self, last_name="LastName", first_name = "FirstName"):
        self.last_name = last_name
        self.first_name = first_name

class Group(list):
    def __init__(self, students = []):
        self = students

    def getKey(self):
        key = ""
        for s in self:
            if len(key) != 0:
                key += "_"
            key += s.last_name
        return key

    def findArchive(self, path, extension):
        """
        Return: archive_path , is_name_acceptable, status_msg

        Perform the following operation:
        1. Test if there is an archive with a valid name in folder 'path'
        2. If no available names are found, list all the archive files and
           request user choice to see if it's valid
        3. If user chooses an archive name manually, it asks if the name is
           valid
        """
        if path[-1] != "/":
            path += "/"
        key = self.getKey()
        candidates = [path + key + extension, path + key.lower() + extension]
        for c in candidates:
            if os.path.exists(c):
                return c , True, ""
        ls_result = systemCall("find {:} -name \"*{:}\"".format(path, extension))
        msg = "Failed to find file with default name, is one of the following file valid?\n"
        msg += "Default names were: " + str(candidates) + "\n"
        msg += "-> answer 'n' if no file is valid\n"
        file_options = ls_result[1].split("\n")
        options = []
        for i in range(len(file_options)):
            msg += "{:2d}: {:}\n".format(i, file_options[i])
            options += [str(i)]
        choice = prompt(msg, options + ["n"])
        if choice == "n":
            return  None, False, "No archive file received"
        choice_idx = int(choice)
        archive_file = file_options[choice_idx]
        error_msg = "Expecting '{:}', received '{:}'".format(candidates[0], archive_file)
        acceptable = question("Is archive name acceptable?")
        return archive_file, acceptable, error_msg

    def extractArchive(self,archive_path):
        """
        return: path_to_extracted_folder, is_format_conform, status_msg
        """
        # Check archive content, should contain all files under a single folder
        # Keeping only first pattern
        exclude_pattern="--exclude=\"[^\.]*/*\" --exclude=\".*/*/*\""
        view_result = systemCall("tar {:} -tvzf {:}".format(exclude_pattern, archive_path))
        invalid_format = False
        status_msg = ""
        if view_result[0] != 0:
            invalid_format = True
            status_msg += "Invalid format"
            view_result = systemCall("tar {:} -tvf {:}".format(exclude_pattern, archive_path))
            if view_result[0] != 0:
                return None, False, "Unable to extract archive content automatically"
        archive_folder = os.path.dirname(archive_path)
        dst_extract = archive_folder
        dst_folder = archive_folder + "/" + self.getKey()
        nb_lines = len(view_result[1].split("\n"))
        if nb_lines != 1:
            if len(status_msg) > 0:
                status_msg += " AND "
            status_msg += "All files should be in a single folder"
            dst_extract = dst_folder
            os.mkdir(dst_folder)
        else:
            #TODO: check folder name
            dst_folder = archive_folder + "/" + view_result[1].split(" ")[-1]
        cmd = "tar -C {:} -xf {:}".format(dst_extract, archive_path)
        extract_result = systemCall(cmd)
        if extract_result[0] != 0:
            return None, False, "Failed to extract archive: {:}".format(extract_result[2])
        return dst_folder, len(status_msg) == 0, status_msg

class GroupCollection(list):
    def __init__(self):
        self = []

    """
    Path should be a json file with an array of array of students
    """
    def load(self, path):
        with open(path) as f:
            val = json.load(f)
            if not isinstance(val, list):
                raise RuntimeError("GroupCollection should be an array in file {:}".format(path))
            self.clear()
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
                self.append(g)

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
                line += " see *{:}".format(msg_index)
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
    return sys.stdin.readline().strip().lower()

def askFloat(msg):
    print(msg)
    print("> ", end='')
    sys.stdout.flush()
    return float(sys.stdin.readline().strip().lower())

def checkNode(node):
    result = prompt("Is: '{:}' ok? (y(es), n(o), p(artially)".format(node.name), ['y','n','p'])
    if (result == 'y'):
        node.points = node.max_points
    else:
        if (result == 'p'):
            node.points = askFloat("How many points? (max_points={:})".format(node.max_points))
        node.msg = freeTextQuestion("What is the problem?")

def systemCall(cmd):
    proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr= subprocess.PIPE, shell=True)
    stdout, stderr = proc.communicate()
    return [proc.returncode, stdout.decode("utf8").strip(), stderr.decode("utf8").strip()]

def approxEq(d1, d2, epsilon = 10**-6):
    return abs(d1 -d2) <= epsilon



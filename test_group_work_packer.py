from group_work_packer import *
import unittest

class TestStudent(unittest.TestCase):
    def test_constructor_lowercase(self):
        s = Student("muray", "bill")
        self.assertEqual("MURAY", s.last_name)
        self.assertEqual("Bill", s.first_name)

    def test_constructor_uppercase(self):
        s = Student("MURAY", "BILL")
        self.assertEqual("MURAY", s.last_name)
        self.assertEqual("Bill", s.first_name)

    def test_constructor_hyphen(self):
        s = Student("mori-son", "paul-henry")
        self.assertEqual("MORI-SON", s.last_name)
        self.assertEqual("Paul-Henry", s.first_name)

    def test_constructor_space(self):
        s = Student("mori son", "paul henry")
        self.assertEqual("MORI SON", s.last_name)
        self.assertEqual("Paul Henry", s.first_name)

class TestGroup(unittest.TestCase):
    def test_group_getkey_1student(self):
        g = Group([Student("smith","john")])
        self.assertEqual("SMITH.John", g.getKey())

    def test_group_getkey_2student(self):
        g = Group([Student("smith","john"), Student("sixpack","joe")])
        self.assertEqual("SIXPACK.Joe_SMITH.John", g.getKey())

    def test_group_getkey_4student(self):
        g = Group([Student("howard","ted"),
                   Student("davy-jones","charlie"),
                   Student("smith","john"),
                   Student("sixpack","joe")])
        self.assertEqual("DC_HT_SJ_SJ", g.getKey())

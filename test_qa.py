import json
import unittest

from qa import answer_from_facts
from train import DATA, make_examples


class QuestionAnsweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(DATA.read_text())["presidents"]

    def answer(self, question):
        return answer_from_facts(question, self.records)

    def test_predecessor_wordings(self):
        self.assertEqual(self.answer("Who served before Pranab Mukherjee?"), "Pratibha Patil.")
        self.assertEqual(self.answer("Who preceded Pranab Mukherjee?"), "Pratibha Patil.")
        self.assertEqual(self.answer("Who was Pranab Mukherjee's predecessor?"), "Pratibha Patil.")
        self.assertEqual(self.answer("Who served 2 terms before Pranab Mukherjee?"), "A. P. J. Abdul Kalam.")
        self.assertEqual(self.answer("Who served two presidents before Pranab Mukherjee?"), "A. P. J. Abdul Kalam.")

    def test_successor_and_term(self):
        self.assertEqual(self.answer("Who succeeded Pranab Mukherjee?"), "Ram Nath Kovind.")
        self.assertEqual(self.answer("Who served 2 terms after Pratibha Patil?"), "Ram Nath Kovind.")
        self.assertEqual(self.answer("What was A. P. J. Abdul Kalam's term of office?"), "25 July 2002 to 25 July 2007.")

    def test_relative_query_boundaries(self):
        self.assertEqual(
            self.answer("Who served 2 terms before Rajendra Prasad?"),
            "There is no term that far before this one.",
        )
        self.assertEqual(
            self.answer("Who served after Droupadi Murmu?"),
            "There is no term that far after this one.",
        )

    def test_date_queries(self):
        self.assertEqual(self.answer("Who became president on 25 July 2017?"), "Ram Nath Kovind.")
        self.assertEqual(self.answer("Who was president on 1 January 2005?"), "A. P. J. Abdul Kalam.")

    def test_summary_queries(self):
        self.assertEqual(self.answer("Who was the first president of India?"), "Rajendra Prasad.")
        self.assertEqual(self.answer("Who is the current president of India?"), "Droupadi Murmu.")
        self.assertEqual(self.answer("How many elected presidents are listed?"), "15.")
        self.assertEqual(self.answer("Who was the 11th president?"), "A. P. J. Abdul Kalam.")

    def test_unknown_question_is_not_guessed(self):
        self.assertIsNone(self.answer("What was the president's favourite food?"))

    def test_training_examples_fit_context(self):
        self.assertLessEqual(max(map(len, make_examples(self.records))), 129)


if __name__ == "__main__":
    unittest.main()


from shared.utils.re_utils import remove_apostrophes, remove_dollar, remove_parentheses


class TestReUtils:
    def test_remove_dollar(self):
        assert remove_dollar("$300.4") == "300.4"
        assert remove_dollar("$1.40") == "1.40"

    def test_remove_parentheses(self):
        assert remove_parentheses("(Hel") == "Hel"
        assert remove_parentheses("(Hello)") == "Hello"

    def test_remove_apostrophes(self):
        assert remove_apostrophes("'") == ""
        assert remove_apostrophes("Hel'lo") == "Hello"

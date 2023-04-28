import re


def remove_dollar(price: str):
    dollar_pattern = re.compile("\$")
    return re.sub(dollar_pattern, "", price)


def remove_parentheses(value: str):
    parentheses_pattern = re.compile("\(|\)")
    return re.sub(parentheses_pattern, "", value)


def remove_apostrophes(value: str):
    apostrophe_pattern = re.compile("'")
    return re.sub(apostrophe_pattern, "", value)

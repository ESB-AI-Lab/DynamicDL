import re

regex_pattern = "(.*) (.*) (.*) (.*)"  # Using a non-greedy match to capture each .* independently
text = "Abyssinian_100 1 1 1"

matches = re.findall(regex_pattern, text)

print(matches)
import re
from re import Match


aufzaehlung_regexp = re.compile(r'\b((ers|zwei|drit|vier|fünf|sechs|sieb|ach|neun|zehn|elf|zwölf)(ter|tes))\b')
zahl_start_regexp = re.compile(r'\b(null|ein|zwei|drei|vier|fünf|sech|sieb|acht|neun|zehn|elf|zwölf|zwan|hundert|tausend)(\w*)')
# 1 - 2
zahl_zehner_regexp1_str = '(?:und)?(zwanzig|dreißig|vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig)'
# 3 -9
zahl_zehner_regexp2_str = '((zehn)|({0}))'.format(zahl_zehner_regexp1_str)
# 1
zahl_regexp_1 = re.compile(r'(emillionen|tausend|hundert)|({0})'.format(zahl_zehner_regexp1_str))
# 2
zahl_regexp_2 = re.compile(r'((millionen|tausend|hundert)|({0}))'.format(zahl_zehner_regexp1_str))
# 3 -> 9
zahl_regexp_3 = re.compile(r'((millionen|tausend|hundert)|({0}))'.format(zahl_zehner_regexp2_str))

num_map = {
    'null': 0,
    'ein': 1,
    'ers': 1,
    'zwei': 2,
    'drei': 3,
    'drit': 3,
    'vier': 4,
    'fünf': 5,
    'sech': 6,
    'sechs': 6,
    'sieb': 7,
    'acht': 8,
    'ach': 8,
    'neun': 9,
    'zehn': 10,
    'elf': 11,
    'zwölf': 12,
    'zwan': 2,
    'hundert': 100,
    'tausend': 1000,
    'millionen': 1000000
}


def extract_number2(num: int, second_match):
    return ''


def extract_number(text: str, first_match):
    # first_match matches the hole number and start() must be 0
    prefix = first_match.group(1)
    postfix = first_match.group(2)

    print(f'text: {text}, prefix: {prefix}, postfix: {postfix}')

    num = num_map[prefix]

    if not postfix:
        if not ((prefix == 'ein') or (prefix == 'sech')):
            return f'{num:d}'
    elif postfix == 's':
        if (prefix == 'ein') or (prefix == 'sech'):
            return f'{num:d}'
    elif prefix == 'ein':
        second_match = zahl_regexp_1.search(postfix)

        if second_match:
            return extract_number2(num, second_match)
    elif prefix == 'zwei':
        second_match = zahl_regexp_1.search(postfix)

        if second_match:
            return extract_number2(num, second_match)

    return ''


def get_number(text: str):
    first_match = zahl_start_regexp.search(text)

    if first_match:
        number = extract_number(text, first_match)

        if number:
            return number, 'zahl', text[first_match.end():]

    return '', 'text', text  # number string, type, pending text


def main():
    n1 = get_number('eins')
    print(f'{n1}')
    n2 = get_number('zwei')
    print(f'{n2}')


if __name__ == '__main__':
    main()

import re
from re import Match


valid_number_chars = '[abcdefghilnrstuvwzßöü]'
aufzaehlung_re = re.compile(r'(.*)(ter|tes|tens)$')
zahl_start_re = re.compile(r'\b(null|ers|ein|zwei|drei|drit|vier|fünf|sech|sieb|acht|ach|neun|zehn|elf|zwölf|zwan|hundert|tausend)({0}*)'.format(valid_number_chars))
einer_re_str = 'ein|zwei|drei|vier|fünf|sech|sieb|acht|neun|zehn|elf|zwölf|zwan'
zehner_re_str = '(?:und)?zwanzig|dreißig|vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig'
zahl_re = re.compile(r'((tausend|hundert|ßig|zig)|({0})|({1}))({2}*)'.format(zehner_re_str, einer_re_str, valid_number_chars))

num_map = {
    'null': 0,
    'ein': 1,
    'ers': 1,
    'eins': 1,
    'zwei': 2,
    'drei': 3,
    'drit': 3,
    'vier': 4,
    'fünf': 5,
    'sech': 6,
    'sechs': 6,
    'sieb': 7,
    'sieben': 7,
    'ach': 8,
    'acht': 8,
    'neun': 9,
    'zehn': 10,
    'zig': 10,
    'ßig': 10,
    'elf': 11,
    'zwölf': 12,
    'zwan': 2,
    'hundert': 100,
    'tausend': 1000
}

num_multipier_map = {
    'null': 0,
    'zig': 10,
    'ßig': 10,
    'hundert': 100,
    'tausend': 1000
}

num_add_map = {
    'null': 0,
    'zwanzig': 20,
    'undzwanzig': 20,
    'dreißig': 30,
    'unddreißig': 30,
    'vierzig': 40,
    'undvierzig': 40,
    'fünfzig': 50,
    'undfünfzig': 50,
    'sechzig': 60,
    'undsechzig': 60,
    'siebzig': 70,
    'undsiebzig': 70,
    'achtzig': 80,
    'undachtzig': 80,
    'neunzig': 90,
    'undneunzig': 90
}

postfix_to_ignore = ['s', 'en']


def extract_number2(num: int, second_match):
    g1 = second_match.group(1)
    postfix = second_match.group(len(second_match.groups()))

    multipier = num_multipier_map.get(g1)

    if multipier:
        if postfix:
            postfix_match = zahl_re.search(postfix)

            if postfix_match:
                return extract_number2(num * multipier, postfix_match)

        return num * multipier

    summand = num_add_map.get(g1)

    if summand:
        if postfix:
            postfix_match = zahl_re.search(postfix)

            if postfix_match:
                return extract_number2(num + summand, postfix_match)

        if (num > 10) and (num < 100):
            return (num * 100) + summand

        return num + summand

    m = num_map.get(second_match.group(1))

    if m:
        if (m < 20) and (m != 10):
            if postfix.startswith("hundert") or postfix.startswith("shundert") or postfix.startswith("enhundert"):
                num += (100 * m)
                postfix = postfix[7 + postfix.index('hundert'):]
            elif (num < 100) and (postfix.startswith("tausend") or postfix.startswith("stausend") or postfix.startswith("etausend")):
                num += (1000 * m)
                postfix = postfix[7 + postfix.index('tausend'):]
            else:
                if (num > 10) and (num < 100):
                    num = (num * 100) + m
                else:
                    num += m
        elif m < 100:
            num += m
        else:
            num *= m

        if postfix and (not postfix in postfix_to_ignore):
            postfix_match = zahl_re.search(postfix)

            if postfix_match:
                return extract_number2(num, postfix_match)

        return num

    return None


def extract_number(text: str, first_match):
    # first_match matches the hole number and start() must be 0
    prefix = first_match.group(1)
    postfix = first_match.group(2) if (len(first_match.groups()) > 1) else ''
    print(f'Number Match: Prefix: {prefix}, Postfix: {postfix}')
    num = num_map.get(prefix)

    if not postfix:
        if (prefix != 'ein') and (prefix != 'sech'):
            return num
    else:
        if postfix == 's':
            if (prefix == 'ein') or (prefix == 'sech'):
                return num
        elif postfix == 'en':
            if prefix == 'sieb':
                return num
        else:
            second_match = zahl_re.search(postfix)

            if second_match:
                return extract_number2(num, second_match)

    return None


def get_number(text: str):
    first_match = zahl_start_re.search(text)

    if first_match:
        if text == f'{first_match.group(1)}{first_match.group(2)}':
            number = extract_number(text, first_match)

            if number:
                return number, 'zahl', text[first_match.end():]

    return '', 'text', text  # number string, type, pending text


def main():
    print(zahl_re.pattern)
    n1 = get_number('eins')
    print(f'{n1}')
    n2 = get_number('zwei')
    print(f'{n2}')
    n13 = get_number('dreizehn')
    print(f'{n13}')
    noNumber = get_number('einsprechen')
    print(f'noNumber: {noNumber}')


if __name__ == '__main__':
    main()

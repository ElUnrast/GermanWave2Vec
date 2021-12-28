import difflib


def html_diff(text, n_text):
    """
    http://stackoverflow.com/a/788780
    Unify operations between two compared strings seqm is a difflib.
    SequenceMatcher instance whose a & b are strings
    """
    output = []
    dist = 0
    output.append('<p style="font-size:100%;">')

    try:
        seqm = difflib.SequenceMatcher(None, text, n_text)

        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.append(seqm.a[a0:a1])
            elif opcode == 'insert':
                if (b1-b0) > 2:
                    dist += 100000 * (b1-b0)
                else:
                    dist += 10000 * (b1-b0)

                output.append("<font color=red>^" + seqm.b[b0:b1] + "</font>")
            elif opcode == 'delete':
                if (a1-a0) > 2:
                    dist += 100000 * (a1-a0)
                else:
                    dist += 10000 * (a1-a0)

                output.append("<font color=blue>^" + seqm.a[a0:a1] + "</font>")
            elif opcode == 'replace':
                # seqm.a[a0:a1] -> seqm.b[b0:b1]
                if (b1-b0) > 2:
                    dist += 100000 * (b1-b0)
                else:
                    dist += 10000 * (b1-b0)

                output.append("<font color=green>^" + seqm.b[b0:b1] + "</font>")
            else:
                raise RuntimeError("unexpected opcode")
    except TypeError:
        print(f'ERROR in {text}')

    output.append('</p>')
    return ''.join(output), dist

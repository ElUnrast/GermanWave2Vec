import re
from typing import List


class SpeechEventHandler():
    def handle_speech_event(self, text: str):
        pass

    def append_formatted_text(self, text: str):
        pass

    def set_bold(self, b: bool):
        pass

    def set_italic(self, b: bool):
        pass


class SpeechComanndProcessor():
    def __init__(
        self,
        processor_name: str,
        start_command: str,
        stop_command: str,
        trigger_words: List[str],
        speech_event_handler: SpeechEventHandler,
        active: bool = False
    ):
        self.name = processor_name
        self.cmd_regexp_map = {}
        self.speech_event_handler = speech_event_handler
        self.start_cmd = start_command
        self.cmd_regexp_map[start_command] = re.compile(r'\b{0}\b'.format(start_command))
        self.stop_cmd = stop_command
        self.cmd_regexp_map[stop_command] = re.compile(r'\b{0}\b'.format(stop_command))
        self.trigger_words = trigger_words

        for cmd in trigger_words:
            self.cmd_regexp_map[cmd] = re.compile(r'\b{0}\b'.format(cmd))

        self.first_cmd_regexp = re.compile('({0})'.format('|'.join(self.cmd_regexp_map.keys())))
        self.active = active

    def get_pending_command(self):
        if self.active and not self.trigger_words:
            return self.start_cmd

        return None

    def process_cmd(self, cmd, text):
        if text:
            print(f'Writing Command {cmd} as: {text}')
            self.speech_event_handler.handle_speech_event(text)

        return True

    def process(self, text):
        if not text:
            return True

        if self.active and self.get_pending_command():
            return self.process_cmd(cmd=self.get_pending_command(), text=text)

        first_match = self.first_cmd_regexp.search(text)

        if not first_match:
            return False

        cmd = first_match.group()
        prefix = text[0:first_match.start()]
        postfix = text[first_match.end():]

        if not self.active:
            if cmd == self.start_cmd:
                if first_match.start() > 0:
                    self.speech_event_handler.handle_speech_event(prefix)

                self.active = True
                self.speech_event_handler.handle_speech_event(postfix)

                return True
        else:
            print(f'found command: {cmd}')
            if first_match.start() > 0:
                self.speech_event_handler.handle_speech_event(prefix)

            if cmd == self.stop_cmd:
                self.speech_event_handler.handle_speech_event(prefix)
                self.active = False
                self.speech_event_handler.handle_speech_event(postfix)
                return True

            return self.process_cmd(cmd, text=postfix)

        return False


class SpeechComanndSatzzeichenProcessor(SpeechComanndProcessor):
    satzzeichen = {
        'punkt': '.',
        'komma': ',',
        'komm ma': ',',
        'komm mal': ',',
        'kommer': ',',
        'komm mal': ',',
        'fragezeichen': '?',
        'ausrufezeichen': '!',
        'bindestrich': '-',
        'minus': '-',
        'plus': ' + ',
        'gleich zeichen': ' = ',
        'semikolon': ';',
        'anführungszeichen unten': '»',
        'anführungszeichen unten': '«',
        'klammer auf': '(',
        'klammer zu': ')',
        'runde klammer auf': '(',
        'runde klammer zu': ')',
        'eckige klammer auf': '[',
        'eckige klammer zu': '[',
        'geschweifte klammer auf': '{',
        'geschweifte klammer zu': '}',
        'zeilenumbruch': '\n',  # '<br/>',
        'neue zeile': '\n',  # '<br/>',
        'absatz': '\n\n'  # <p/>'
    }

    def __init__(self, speech_event_handler, active: bool = True):
        super().__init__(
            processor_name='Satzzeichen',
            start_command='satzzeichen diktieren',
            stop_command='satzzeichen aus',
            trigger_words=SpeechComanndSatzzeichenProcessor.satzzeichen.keys(),
            speech_event_handler=speech_event_handler,
            active=active
        )

    def process_cmd(self, cmd, text):
        mapped_satzzeichen = SpeechComanndSatzzeichenProcessor.satzzeichen[cmd]
        print(f'Mapped {cmd} -> {mapped_satzzeichen}')
        self.speech_event_handler.append_formatted_text(mapped_satzzeichen)
        return super().process_cmd(cmd, text)


class SpeechComanndFormatProcessor(SpeechComanndProcessor):
    formate = {
        'fett aus': '</b>',
        'fett ende': '</b>',
        'fett': '<b>',
        'kursiv aus': '</l>',
        'kursiv ende': '</l>',
        'kursiv': '<l>',
    }

    def __init__(self, speech_event_handler, active: bool = True):
        super().__init__(
            processor_name='Formatierung',
            start_command='formatierung ein',
            stop_command='formatierung aus',
            trigger_words=SpeechComanndFormatProcessor.formate.keys(),
            speech_event_handler=speech_event_handler,
            active=active
        )

    def process_cmd(self, cmd, text):
        mapped_format = SpeechComanndFormatProcessor.formate[cmd]
        print(f'Mapped {cmd} -> {mapped_format}')

        if '<b>' == mapped_format:
            self.speech_event_handler.set_bold(True)
        elif '</b>' == mapped_format:
            self.speech_event_handler.set_bold(False)
        elif '<l>' == mapped_format:
            self.speech_event_handler.set_italic(True)
        elif '</l>' == mapped_format:
            self.speech_event_handler.set_italic(False)

        return super().process_cmd(cmd, text)


class SpeechComanndBuchstabierenProcessor(SpeechComanndProcessor):
    def __init__(self, speech_event_handler, active: bool = False):
        super().__init__(
            processor_name='Buchstabieren',
            start_command='ich buchstabiere',
            stop_command='buchstabieren aus',
            trigger_words=[],
            speech_event_handler=speech_event_handler,
            active=active
        )

    def process_cmd(self, cmd, text):
        stop_match = self.cmd_regexp_map[self.stop_cmd].search(text)

        if stop_match:
            self._append_text(text[0:stop_match.start()])
            self.active = False
            self.speech_event_handler.handle_speech_event(text[stop_match.end():])
            return True

        return self._append_text(text)

    def _append_text(self, text: str):
        if text:
            wort = ''.join([w[0] for w in text.split(' ')])
            self.speech_event_handler.append_formatted_text(wort)

        return True


class SpeechComanndDatumProcessor(SpeechComanndProcessor):
    def __init__(self, speech_event_handler, active: bool = True):
        super().__init__(
            processor_name='Buchstabieren',
            start_command='ich buchstabiere',
            stop_command='buchstabieren aus',
            trigger_words=[],
            speech_event_handler=speech_event_handler,
            active=active
        )

    def process_cmd(self, cmd, text):
        stop_match = self.cmd_regexp_map[self.stop_cmd].search(text)

        if stop_match:
            self._append_text(text[0:stop_match.start()])
            self.active = False
            self.speech_event_handler.handle_speech_event(text[stop_match.end():])
            return True

        return self._append_text(text)

    def _append_text(self, text: str):
        pass  # TODO


class SpeechComanndZahlProcessor(SpeechComanndProcessor):
    aufzaehlung_regexp = re.compile(r'\b((ers|zwei|drit|vier|fünf|sechs|sieb|ach|neun|zehn|elf|zwölf)(ter|tes))\b'.format())
    zahl_start_regexp = re.compile(r'\b((ein|zwei|drei|vier|fünf|sech|sieb|acht|neun|zehn|elf|zwölf|zwan|hundert|tausend))'.format())

    def __init__(self, speech_event_handler, active: bool = True):
        pass  # TODO

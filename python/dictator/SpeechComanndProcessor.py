import re
from typing import List
from NumberConverter import zahl_start_re, aufzaehlung_re, extract_number


class SpeechEventHandler():
    def handle_speech_event(self, text: str):
        pass

    def is_formated_text_at_block_start(self):
        pass

    def get_last_proccessor(self):
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
        trigger,  # List of words or Dictionary cmd -> regexp
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
        self.trigger = trigger

        if trigger:
            if isinstance(trigger, dict):
                self.cmd_regexp_map.update(trigger)
            else:
                print(f'{type(trigger)}')
                for cmd in trigger:
                    self.cmd_regexp_map[cmd] = re.compile(r'\b{0}\b'.format(cmd))

        pattern_list = []
        zahl_start_re.pattern

        for p in self.cmd_regexp_map.values():
            pattern_list.append(p.pattern)

        self.first_cmd_regexp = re.compile('({0})'.format('|'.join(pattern_list)))
        self.active = active

    def get_pending_command(self):
        if self.active and not self.trigger:
            return self.start_cmd

        return None

    def process_cmd(self, match, cmd, text):
        if text:
            print(f'Writing Command {cmd} as: {text}')
            self.speech_event_handler.handle_speech_event(text)

        return True

    def find_first_match(self, text):
        first_idx = None
        first_cmd = None
        first_match = None

        for cmd, pattern in self.cmd_regexp_map.items():
            match = pattern.search(text)

            if match:
                if (not first_idx) or match.start() < first_idx:
                    first_idx = match.start()
                    first_match = match
                    first_cmd = cmd

                    if first_idx == 0:
                        break

        return first_cmd, first_match

    def process(self, text):
        if not text:
            return True

        cmd, first_match = self.find_first_match(text)

        if self.active and self.get_pending_command():
            return self.process_cmd(match=first_match, cmd=self.get_pending_command(), text=text)

        if not first_match:
            return False

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

            return self.process_cmd(match=first_match, cmd=cmd, text=postfix)

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
            trigger=SpeechComanndSatzzeichenProcessor.satzzeichen.keys(),
            speech_event_handler=speech_event_handler,
            active=active
        )

    def process_cmd(self, match, cmd, text):
        mapped_satzzeichen = SpeechComanndSatzzeichenProcessor.satzzeichen[cmd]
        print(f'Mapped {cmd} -> {mapped_satzzeichen}')
        self.speech_event_handler.append_formatted_text(mapped_satzzeichen)
        return super().process_cmd(match, cmd, text)


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
            trigger=SpeechComanndFormatProcessor.formate.keys(),
            speech_event_handler=speech_event_handler,
            active=active
        )

    def process_cmd(self, match, cmd, text):
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

        return super().process_cmd(match, cmd, text)


class SpeechComanndBuchstabierenProcessor(SpeechComanndProcessor):
    def __init__(self, speech_event_handler, active: bool = False):
        super().__init__(
            processor_name='Buchstabieren',
            start_command='ich buchstabiere',
            stop_command='buchstabieren aus',
            trigger=[],
            speech_event_handler=speech_event_handler,
            active=active
        )

    def process_cmd(self, match, cmd, text):
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


class SpeechComanndZahlProcessor(SpeechComanndProcessor):
    def __init__(self, speech_event_handler, active: bool = True):
        trigger = {}
        trigger['zahl'] = zahl_start_re
        trigger['datum'] = re.compile(r'\b{0}\b'.format('datum'))
        trigger['datum aus'] = re.compile(r'\b{0}\b'.format('datum aus'))
        super().__init__(
            processor_name='Zahlen umwandeln',
            start_command='zahlen umwandeln',
            stop_command='zahlen umwandeln aus',
            trigger=trigger,
            speech_event_handler=speech_event_handler,
            active=active
        )
        self.datum_erkennen = True

    def process_cmd(self, match, cmd, text):
        if 'datum' == cmd:
            self.datum_erkennen = True
        elif 'datum_aus' == cmd:
            self.datum_erkennen = False
        else:
            typ = 'zahl'
            matched_text = match.group()
            print(f'Potentielle Zahl erkannt in: {matched_text}')
            aufzaehlungMatcher = aufzaehlung_re.search(matched_text)

            if aufzaehlungMatcher:
                matched_text = aufzaehlungMatcher.group(1)
                match = zahl_start_re.search(matched_text)

                if not match:
                    return False

                typ = 'aufzaehlung'

            mapped_number = extract_number(text, match)

            if mapped_number:
                print('Zahlen Typ: {typ}')

                if 'aufzaehlung' == typ:
                    formatted_text = f'{mapped_number:d}.'
                else:
                    formatted_text = f'{mapped_number:d}'

                prepend_space = not self.speech_event_handler.is_formated_text_at_block_start()

                if prepend_space and self.speech_event_handler.get_last_proccessor():
                    if 'Zahlen umwandeln' == self.speech_event_handler.get_last_proccessor().name:
                        prepend_space = False

                if prepend_space:
                    formatted_text = f' {formatted_text}'

                self.speech_event_handler.append_formatted_text(formatted_text)
            else:
                return False

        return super().process_cmd(match, cmd, text)

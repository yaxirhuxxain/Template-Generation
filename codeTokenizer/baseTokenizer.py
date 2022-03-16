import re
import unicodedata
from collections import namedtuple
from typing import List
import six

Position = namedtuple('Position', ['line', 'column'])


class Token(object):
    def __init__(self, value, position=None, doc=None):
        self.value = value
        self.position = position
        self.doc = doc

    def __repr__(self):
        if self.position:
            return '%s "%s" line %d, position %d' % (
                self.__class__.__name__, self.value, self.position[0], self.position[1]
            )
        else:
            return '%s "%s"' % (self.__class__.__name__, self.value)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        raise Exception("Direct comparison not allowed")


class LexerError(Exception):
    pass


class Literal(Token):
    pass


class Integer(Literal):
    pass


class DecimalInteger(Literal):
    pass


class OctalInteger(Integer):
    pass


class BinaryInteger(Integer):
    pass


class HexInteger(Integer):
    pass


class FloatingPoint(Literal):
    pass


class DecimalFloatingPoint(FloatingPoint):
    pass


class HexFloatingPoint(FloatingPoint):
    pass


class Boolean(Literal):
    VALUES = set(["true", "false"])


class Character(Literal):
    pass


class String(Literal):
    pass


class Null(Literal):
    pass


class EndOfInput(Token):
    pass


class Annotation(Token):
    pass


class Identifier(Token):
    pass


class Separator(Token):
    VALUES = set(['(', ')', '{', '}', '[', ']', ';', ',', '.'])


class Operator(Token):
    MAX_LEN = 4
    VALUES = set(['>>>=', '>>=', '<<=', '%=', '^=', '|=', '&=', '/=',
                  '*=', '-=', '+=', '<<', '--', '++', '||', '&&', '!=',
                  '>=', '<=', '==', '%', '^', '|', '&', '/', '*', '-',
                  '+', ':', '?', '~', '!', '<', '>', '=', '...', '->', '::', '#'])

    # '>>>' and '>>' are excluded so that >> becomes two tokens and >>> becomes
    # three. This is done because we can not distinguish the operators >> and
    # >>> from the closing of multipel type parameter/argument lists when
    # lexing. The job of potentially recombining these symbols is left to the
    # parser

    INFIX = set(['||', '&&', '|', '^', '&', '==', '!=', '<', '>', '<=', '>=',
                 '<<', '>>', '>>>', '+', '-', '*', '/', '%'])

    PREFIX = set(['++', '--', '!', '~', '+', '-', '#'])

    POSTFIX = set(['++', '--'])

    ASSIGNMENT = set(['=', '+=', '-=', '*=', '/=', '&=', '|=', '^=', '%=',
                      '<<=', '>>=', '>>>='])

    LAMBDA = set(['->'])

    METHOD_REFERENCE = set(['::', ])

    def is_infix(self):
        return self.value in self.INFIX

    def is_prefix(self):
        return self.value in self.PREFIX

    def is_postfix(self):
        return self.value in self.POSTFIX

    def is_assignment(self):
        return self.value in self.ASSIGNMENT


class Keyword(Token):
    VALUES = None


class Modifier(Keyword):
    VALUES = set(['abstract', 'default', 'final', 'finally', 'native', 'private',
                  'protected', 'public', 'static', 'strictfp', 'synchronized',
                  'transient', 'volatile'])


class BasicType(Keyword):
    VALUES = set(['boolean', 'bool', 'byte', 'char', 'double',
                  'float', 'int', 'long', 'short', 'String'])


class Tokenizer(object):
    IDENT_START_CATEGORIES = set(['Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nl', 'Pc', 'Sc'])

    IDENT_PART_CATEGORIES = set(['Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Mc', 'Mn', 'Nd', 'Nl', 'Pc', 'Sc'])

    def __init__(self, data: str, language: str, grammar: List, ignore_errors=False):
        self.data = data
        self.ignore_errors = ignore_errors
        self.language = language
        self.KEYWORDS = grammar
        self.errors = []
        Keyword.VALUES = set(self.KEYWORDS)

        # Rows and columns both start at 1
        self.current_line = 1
        self.start_of_line = -1

        self.operators = [set() for i in range(0, Operator.MAX_LEN)]

        for v in Operator.VALUES:
            self.operators[len(v) - 1].add(v)

        self.whitespace_consumer = re.compile(r'[^\s]')

        self.doc = None
        self.stringMute = False

    def reset(self):
        self.i = 0
        self.j = 0

    def consume_whitespace(self):
        match = self.whitespace_consumer.search(self.data, self.i + 1)

        if not match:
            self.i = self.length
            return

        i = match.start()

        start_of_line = self.data.rfind('\n', self.i, i)

        if start_of_line != -1:
            self.start_of_line = start_of_line
            self.current_line += self.data.count('\n', self.i, i)

        self.i = i

    def read_string(self):
        delim = self.data[self.i]
        if delim == '\\':
            if self.data[self.i - 1] == '"':
                delim = '"'
            elif self.data[self.i - 1] == "'":
                delim = "'"
            else:
                self.error('unknown character/string literal start')

        state = 0
        j = self.i + 1
        length = self.length
        start = self.i

        if self.stringMute == True:
            last_char = ""
            last_last_char = ""

            if self.data[j] == '"':
                delim = '"'
            elif self.data[j] == "'":
                delim = "'"
            else:
                self.error('unknown character/string literal start')

            j = j + 1  # need to adjust the index for c#
            while True:

                if j >= length:
                    self.error('Unterminated character/string literal')
                    self.stringMute = False
                    break

                if self.data[j] == delim:  # " == "
                    if last_char == self.data[j]:  # "" == ""
                        if last_char == last_last_char:  # """ == """
                            if self.data[j] == self.data[j + 1]:  # """" == """"
                                j += 1  # if four quotations then increment 1 more index
                            self.stringMute = False
                            break
                        elif last_last_char == "\\" and last_char == last_last_char:
                            self.stringMute = False
                            break
                    else:
                        if self.data[j] != self.data[j + 1]:
                            self.stringMute = False
                            break

                last_char = self.data[j]
                last_last_char = self.data[j - 1]

                j += 1
            self.j = j + 1

        else:

            while True:
                if j >= length:
                    self.error('Unterminated character/string literal')
                    break

                if state == 0:
                    if self.data[j] == '\\':
                        state = 1
                    elif self.data[j] == delim:
                        break

                elif state == 1:
                    if self.data[j] in 'btnfru"\'\\':
                        state = 0
                    elif self.data[j] in '0123':
                        state = 2
                    elif self.data[j] in '01234567':
                        state = 3
                    else:
                        # self.error('Illegal escape character', self.data[j]) #pass this check for c#
                        state = 3

                elif state == 2:
                    # Possibly long octal
                    if self.data[j] in '01234567':
                        state = 3
                    elif self.data[j] == '\\':
                        state = 1
                    elif self.data[j] == delim:
                        break

                elif state == 3:
                    state = 0

                    if self.data[j] == '\\':
                        state = 1
                    elif self.data[j] == delim:
                        break

                j += 1

            self.j = j + 1

    def try_operator(self):
        for l in range(min(self.length - self.i, Operator.MAX_LEN), 0, -1):
            if self.data[self.i:self.i + l] in self.operators[l - 1]:
                self.j = self.i + l
                return True
        return False

    def read_comment(self):
        if self.data[self.i + 1] == '/':
            terminator = '\n'
        else:
            terminator = '*/'

        i = self.data.find(terminator, self.i + 2)

        if i == -1:
            self.i = self.length
            return

        i += len(terminator)

        comment = self.data[self.i:i]
        start_of_line = self.data.rfind('\n', self.i, i)

        if start_of_line != -1:
            self.start_of_line = start_of_line
            self.current_line += self.data.count('\n', self.i, i)

        self.i = i

        return comment

    def read_decimal_float_or_integer(self):
        orig_i = self.i
        self.j = self.i

        self.read_decimal_integer()

        if self.j >= len(self.data) or self.data[self.j] not in '.eEfFdD':
            return DecimalInteger

        if self.data[self.j] == '.':
            self.i = self.j + 1
            self.read_decimal_integer()

        if self.j < len(self.data) and self.data[self.j] in 'eE':
            self.j = self.j + 1

            if self.j < len(self.data) and self.data[self.j] in '-+':
                self.j = self.j + 1

            self.i = self.j
            self.read_decimal_integer()

        if self.j < len(self.data) and self.data[self.j] in 'fFdD':
            self.j = self.j + 1

        self.i = orig_i
        return DecimalFloatingPoint

    def read_hex_integer_or_float(self):
        orig_i = self.i
        self.j = self.i + 2

        self.read_hex_integer()

        if self.j >= len(self.data) or self.data[self.j] not in '.pP':
            return HexInteger

        if self.data[self.j] == '.':
            self.j = self.j + 1
            self.read_digits('0123456789abcdefABCDEF')

        if self.j < len(self.data) and self.data[self.j] in 'pP':
            self.j = self.j + 1
        else:
            self.error('Invalid hex float literal')

        if self.j < len(self.data) and self.data[self.j] in '-+':
            self.j = self.j + 1

        self.i = self.j
        self.read_decimal_integer()

        if self.j < len(self.data) and self.data[self.j] in 'fFdD':
            self.j = self.j + 1

        self.i = orig_i
        return HexFloatingPoint

    def read_digits(self, digits):
        tmp_i = 0
        c = None

        while self.j + tmp_i < len(self.data):
            c = self.data[self.j + tmp_i]

            if c in digits:
                self.j += 1 + tmp_i
                tmp_i = 0
            elif c == '_':
                tmp_i += 1
            else:
                break

        if c in 'lL':
            self.j += 1

    def read_decimal_integer(self):
        self.j = self.i
        self.read_digits('0123456789')

    def read_hex_integer(self):
        self.j = self.i + 2
        self.read_digits('0123456789abcdefABCDEF')

    def read_bin_integer(self):
        self.j = self.i + 2
        self.read_digits('01')

    def read_octal_integer(self):
        self.j = self.i + 1
        self.read_digits('01234567')

    def read_integer_or_float(self, c, c_next):
        if c == '0' and c_next in 'xX':
            return self.read_hex_integer_or_float()
        elif c == '0' and c_next in 'bB':
            self.read_bin_integer()
            return BinaryInteger
        elif c == '0' and c_next in '01234567':
            self.read_octal_integer()
            return OctalInteger
        else:
            return self.read_decimal_float_or_integer()

    def try_separator(self):
        if self.data[self.i] in Separator.VALUES:
            self.j = self.i + 1
            return True
        return False

    def decode_data(self):
        # Encodings to try in order
        codecs = ['iso-8859-1', 'utf-8']

        # If data is already unicode don't try to redecode
        if isinstance(self.data, six.text_type):
            return self.data

        for codec in codecs:
            try:
                data = self.data.decode(codec)
                return data
            except UnicodeDecodeError:
                pass

        self.error('Could not decode input data')

    def is_identifier_start(self, c):
        return unicodedata.category(c) in self.IDENT_START_CATEGORIES

    def read_identifier(self):
        self.j = self.i + 1

        while self.j < len(self.data) and unicodedata.category(self.data[self.j]) in self.IDENT_PART_CATEGORIES:
            self.j += 1

        ident = self.data[self.i:self.j]
        if ident in Keyword.VALUES:
            token_type = Keyword

            if ident in BasicType.VALUES:
                token_type = BasicType
            elif ident in Modifier.VALUES:
                token_type = Modifier

        elif ident in Boolean.VALUES:
            token_type = Boolean
        elif ident == 'null':
            token_type = Null
        else:
            token_type = Identifier

        return token_type

    def pre_tokenize(self):
        new_data = list()
        data = self.decode_data()

        if self.language == 'C#' and data.startswith('\ufeff'):
            while data.startswith('\ufeff'):
                data = data[1:]

        i = 0
        j = 0
        length = len(data)

        NONE = 0
        ELIGIBLE = 1
        MARKER_FOUND = 2

        state = NONE

        while j < length:
            if state == NONE:
                j = data.find('\\', j)

                if j == -1:
                    j = length
                    break

                state = ELIGIBLE

            elif state == ELIGIBLE:
                c = data[j]

                if c == 'u':
                    state = MARKER_FOUND
                    new_data.append(data[i:j - 1])
                else:
                    state = NONE

            elif state == MARKER_FOUND:
                c = data[j]

                if c != 'u':
                    try:
                        escape_code = int(data[j:j + 4], 16)
                        new_data.append(six.unichr(escape_code))

                        i = j + 4
                        j = i

                        state = NONE

                        continue
                    except ValueError:
                        if self.language == "C#":  # passing this issue for c#
                            pass
                        else:
                            self.error('Invalid unicode escape', data[j:j + 4])

            j = j + 1

        new_data.append(data[i:])

        self.data = ''.join(new_data)
        self.length = len(self.data)

    def tokenize(self):
        self.reset()

        # Convert unicode escapes
        self.pre_tokenize()

        while self.i < self.length:
            token_type = None

            c = self.data[self.i]
            c_next = None
            startswith = c

            if self.i + 1 < self.length:
                c_next = self.data[self.i + 1]
                startswith = c + c_next

            if c.isspace():
                self.consume_whitespace()
                continue

            elif startswith in ("//", "/*"):
                try:
                    comment = self.read_comment()
                    if comment.startswith("/**"):
                        self.doc = comment
                    continue
                except AttributeError:
                    continue

            elif startswith == '..' and self.try_operator():
                # Ensure we don't mistake a '...' operator as a sequence of
                # three '.' separators. This is done as an optimization instead
                # of moving try_operator higher in the chain because operators
                # aren't as common and try_operator is expensive
                token_type = Operator

            elif c == '.' and c_next and c_next.isdigit():
                token_type = self.read_decimal_float_or_integer()

            elif self.try_separator():
                token_type = Separator


            elif self.language == "C#" and startswith in ("@'", '@"'):
                token_type = String
                self.stringMute = True
                self.read_string()



            elif c in ("'", '"'):
                token_type = String
                self.read_string()


            elif self.language == "C#" and c == '\\' and self.data[self.i - 1] in ('"', "'"):
                token_type = String
                self.read_string()



            elif c == '@':
                token_type = Annotation
                self.j = self.i + 1


            elif c in '0123456789':
                token_type = self.read_integer_or_float(c, c_next)

            elif self.is_identifier_start(c):
                token_type = self.read_identifier()

            elif self.try_operator():
                token_type = Operator


            else:
                self.error('Could not process token', c)
                self.i = self.i + 1
                continue

            position = Position(self.current_line, self.i - self.start_of_line)
            token = token_type(self.data[self.i:self.j], position, self.doc)
            yield token

            if self.doc:
                self.doc = None

            self.i = self.j

    def error(self, message, char=None):
        # Provide additional information in the errors message
        line_start = self.data.rfind('\n', 0, self.i) + 1
        line_end = self.data.find('\n', self.i)
        line = self.data[line_start:line_end].strip()

        line_number = self.current_line

        if not char:
            char = self.data[self.j]

        message = u'%s at "%s", line %s: %s' % (message, char, line_number, line)
        error = LexerError(message)
        self.errors.append(error)

        if not self.ignore_errors:
            raise error


def tokenizer(data, language, grammar, ignore_errors=False):
    if type(grammar) != list:
        raise Exception("Language Grammer should be of type list")

    global KEYWORDS
    KEYWORDS = grammar
    tokenizer = Tokenizer(data, language, grammar, ignore_errors)
    return tokenizer.tokenize()

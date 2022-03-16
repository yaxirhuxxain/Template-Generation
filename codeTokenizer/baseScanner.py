# -*- coding: utf-8 -*-


from .baseTokenizer import tokenizer


class tokenizerSimplifier():
    def __init__(self, language, grammar):

        if type(grammar) != list:
            raise Exception("Language Grammar should be of type list")

        global KEYWORDS
        self.language = language
        KEYWORDS = grammar
        self.Keyword = grammar

        self.Separator = ['(', ')', '{', '}', '[', ']', ';', ',', '.']
        self.Operator = ['>>>=', '>>=', '<<=', '%=', '^=', '|=', '&=', '/=',
                         '*=', '-=', '+=', '<<', '--', '++', '||', '&&', '!=',
                         '>=', '<=', '==', '%', '^', '|', '&', '/', '*', '-',
                         '+', ':', '?', '~', '!', '<', '>', '=', '...', '->', '::']

        self.Literal_Types = ["Integer", "DecimalInteger", "OctalInteger", "BinaryInteger", "HexInteger",
                              "FloatingPoint", "DecimalFloatingPoint", "HexFloatingPoint", "Boolean", "Character",
                              "String", "Null"]

    def lex_to_list_type(self, code=str):
        list_code = []

        tokens_list = list(tokenizer(data=code, language=self.language, grammar=KEYWORDS, ignore_errors=False))
        for token in tokens_list:
            list_code.append([str(token).split()[0], token.value])

        return list_code

    def get_clean_code(self, code=str, ):

        list_code = self.lex_to_list_type(code)

        out_code = ""
        for count, token in enumerate(list_code):
            out_code += " " + token[1]

        return out_code

    def get_simplified_code(self, code=str):
        list_code = self.lex_to_list_type(code)

        out_code = ""
        for count, token in enumerate(list_code):
            if token[0] in self.Literal_Types:
                if token[0] in ["Integer", "DecimalInteger", "OctalInteger", "BinaryInteger", "HexInteger"]:
                    out_code += " IntLiteral"
                elif token[0] in ["FloatingPoint", "DecimalFloatingPoint", "HexFloatingPoint"]:
                    out_code += " FloatLiteral"
                elif token[0] == "Boolean":
                    out_code += " BooleanLiteral"
                elif token[0] == "String":
                    out_code += " StringLiteral"
                elif token[0] == "Null":
                    out_code += " NullLiteral"
                else:
                    out_code += " " + token[0] + "Literal"
            else:
                out_code += " " + token[1]

        return out_code

    def get_code_structured(self, code):
        indent = 0
        closed_block = False
        ident_last = False

        output = list()
        code = self.get_clean_code(code)

        for token in code.split():
            if closed_block:
                closed_block = False
                indent -= 4

                output.append('\n')
                output.append(' ' * indent)
                output.append('}\n')

                if token in self.Literal_Types + self.Keyword:
                    output.append('\n')
                    output.append(' ' * indent)

            if token == '{':
                indent += 4
                output.append(' {\n')
                output.append(' ' * indent)

            elif token == '}':
                closed_block = True

            elif token == ',':
                output.append(' , ')

            elif token in self.Literal_Types + self.Keyword:
                if ident_last:
                    # If the last token was a literla/keyword/identifer put a space in between
                    output.append(' ')
                ident_last = True
                output.append(token)

            elif token in self.Operator:
                output.append(' ' + token + ' ')

            elif token == ';':
                output.append(' ;\n')
                output.append(' ' * indent)

            elif token == '(':
                output.append(' ( ')

            elif token == ')':
                output.append(' ) ')

            else:
                output.append(" " + token)

            ident_last = [item for item in self.Literal_Types + self.Keyword if item == token]

        if closed_block:
            output.append('\n}\n')

        output.append('\n')
        # print(''.join(output))
        return ''.join(output)


def scanner(language, grammar):
    if type(grammar) != list:
        raise Exception("Language Grammer should be of type list")

    if language not in ('C#', 'Java'):
        raise Exception("Language Grammer should be of type list")

    return tokenizerSimplifier(language, grammar)


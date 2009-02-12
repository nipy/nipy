#!/usr/bin/env python
''' Script to do very basic conversion of TRAC to reST format '''

import sys
import re

rst_section_levels = ['*', '=', '-', '~', '^']
heading_re = re.compile('^(=+) +([^=]+) +(=+)')
enumerate_re = re.compile('^( +\\d+\\))')
link_re = re.compile(r'\[+ *(http://[\w.\-~/]+) +(.+?) *\]+')
italic_re = re.compile(r"'''(.+?)'''")
bold_re = re.compile(r"''(.+?)''")
inpre_re = re.compile(r"{{{(.+?)}}}")
outprestart_re = re.compile(r"^ *{{{ *$")
outprestop_re = re.compile(r"^ *}}} *$")


def preformatted_state(line, lines):
    ''' State function for within preformatted blocks '''
    if outprestop_re.match(line):
        lines.append('')
        return standard_state
    lines.append('   ' + line)
    return preformatted_state


def standard_state(line, lines):
    ''' State function for within normal text '''
    # beginning preformat block?
    if outprestart_re.match(line):
        lines.append('::')
        return preformatted_state
    # Heading
    hmatch = heading_re.match(line)
    if hmatch:
        eq1, heading, eq2 = hmatch.groups()
        if len(eq1) == len(eq2):
            lines.append(heading)
            lines.append(rst_section_levels[len(eq1)] * len(heading))
            return standard_state
    if line.startswith(' *'):
        line = line[1:]
    ematch = enumerate_re.match(line)
    if ematch:
        start = ematch.groups()[0]
        line = '#.' + line[len(start):]
    line = link_re.sub(r'`\2 <\1>`_', line)
    line = italic_re.sub(r'*\1*', line)
    line = bold_re.sub(r'**\1**', line)
    line = inpre_re.sub(r'``\1``', line)
    lines.append(line)
    return standard_state


def trac2rst(linesource):
    ''' Process trac line source 

    A small simple finite state machine

    >>> lines = ['Hello', '= Heading1 =', '=Heading2=', '== Heading 3 ==']
    >>> trac2rst(lines)
    ['Hello', 'Heading1', '========', '=Heading2=', 'Heading 3', '---------']
    >>> trac2rst([' * bullet point'])
    ['* bullet point']
    >>> trac2rst([' 33 not numbered'])
    [' 33 not numbered']
    >>> trac2rst([' 33) numbered'])
    ['#. numbered']
    >>> trac2rst(['some text then [http://www.python.org/doc a link], then text'])
    ['some text then `a link <http://www.python.org/doc>`_, then text']
    >>> line = 'text [http://www.python.org python] ' + \
               'text [http://www.scipy.org scipy] '
    >>> trac2rst([line])
    ['text `python <http://www.python.org>`_ text `scipy <http://www.scipy.org>`_']
    >>> # The next line conceals the triple quotes from the docstring parser
    >>> trac2rst([r"Some %sitalic%s text, then %smore%s" % (("'"*3,) * 4)])
    ['Some *italic* text, then *more*']
    >>> trac2rst([r"Some ''bold'' text, then ''more''"])
    ['Some **bold** text, then **more**']
    >>> # inline preformatted text
    >>> trac2rst(['here is some {{{preformatted text}}} (end)'])
    ['here is some ``preformatted text`` (end)']
    >>> # multiline preformatted
    >>> trac2rst(['','{{{','= preformatted =', ' * text', '}}}'])
    ['', '::', '   = preformatted =', '    * text', '']
    '''
    lines = []
    processor = standard_state
    for line in linesource:
        line = line.rstrip()
        processor = processor(line, lines)
    return lines


if __name__ == '__main__':
    try:
        infile = sys.argv[1]
    except IndexError:
        raise OSError('Need input file')
    lines = trac2rst(file(infile))
    print '\n'.join(lines)

"""Tiny OMML (Office Math) builder producing consistently formatted equations."""
from xml.sax.saxutils import escape

RPR = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
CTRL = f'<m:ctrlPr>{RPR}</m:ctrlPr>'


def r(text, plain=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ''
    return f'<m:r>{sty}{RPR}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def p(text):
    """Upright (non-italic) text: numbers, operators, function names, words."""
    return r(text, plain=True)


def cat(*xs):
    return "".join(xs)


def sub(base, s):
    return f'<m:sSub><m:sSubPr>{CTRL}</m:sSubPr><m:e>{base}</m:e><m:sub>{s}</m:sub></m:sSub>'


def sup(base, s):
    return f'<m:sSup><m:sSupPr>{CTRL}</m:sSupPr><m:e>{base}</m:e><m:sup>{s}</m:sup></m:sSup>'


def subsup(base, s, u):
    return (f'<m:sSubSup><m:sSubSupPr>{CTRL}</m:sSubSupPr><m:e>{base}</m:e><m:sub>{s}</m:sub>'
            f'<m:sup>{u}</m:sup></m:sSubSup>')


def frac(n, d, lin=False):
    t = '<m:type m:val="lin"/>' if lin else ''
    return f'<m:f><m:fPr>{t}{CTRL}</m:fPr><m:num>{n}</m:num><m:den>{d}</m:den></m:f>'


def acc(x, ch):
    return f'<m:acc><m:accPr><m:chr m:val="{ch}"/>{CTRL}</m:accPr><m:e>{x}</m:e></m:acc>'


def bar(x):
    return f'<m:bar><m:barPr><m:pos m:val="top"/>{CTRL}</m:barPr><m:e>{x}</m:e></m:bar>'


def hat(x):
    return acc(x, "̂")


def d(x, beg="(", end=")"):
    return (f'<m:d><m:dPr><m:begChr m:val="{escape(beg)}"/><m:endChr m:val="{escape(end)}"/>{CTRL}</m:dPr>'
            f'<m:e>{x}</m:e></m:d>')


def nary(op, lo, hi, body):
    lo_x = f'<m:sub>{lo}</m:sub>'
    hi_x = f'<m:sup>{hi}</m:sup>'
    hide = ''
    if not hi:
        hide = '<m:supHide m:val="1"/>'
    return (f'<m:nary><m:naryPr><m:chr m:val="{op}"/><m:limLoc m:val="undOvr"/>{hide}{CTRL}</m:naryPr>'
            f'{lo_x}{hi_x}<m:e>{body}</m:e></m:nary>')


def func(name, arg):
    return (f'<m:func><m:funcPr>{CTRL}</m:funcPr><m:fName>{p(name)}</m:fName>'
            f'<m:e>{arg}</m:e></m:func>')


def eqarr(*rows):
    return f'<m:eqArr><m:eqArrPr>{CTRL}</m:eqArrPr>' + "".join(f'<m:e>{x}</m:e>' for x in rows) + '</m:eqArr>'


def omath(x):
    return f'<m:oMath>{x}</m:oMath>'


def eq_paragraph(math_xml, number):
    """Centered equation with right-aligned number (tab stops on a 9617-twip text width)."""
    return ('<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<w:pPr><w:tabs><w:tab w:val="center" w:pos="4808"/><w:tab w:val="right" w:pos="9600"/></w:tabs>'
            '<w:spacing w:before="80" w:after="80" w:line="300" w:lineRule="auto"/></w:pPr>'
            '<w:r><w:tab/></w:r>' + omath(math_xml) +
            f'<w:r><w:tab/><w:t>({number})</w:t></w:r></w:p>')


# shorthand symbols
def v(name, s=None):
    """italic variable with optional subscript (italic)"""
    return sub(r(name), r(s)) if s is not None else r(name)

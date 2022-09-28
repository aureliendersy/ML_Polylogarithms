"""Utility routines for the link with the Wolfram Kernel
Also contains the relevant routines for adding data generated with scripts in Mathematica (for instance the symbol)"""

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export
from ..utils import timeout
from sympy import parse_expr
from src.envs.char_sp import UnknownSymPyOperator, CharSPEnvironment
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from logging import getLogger
import sympy as sp
import time

logger = getLogger()


def start_wolfram_session(kernel_path=None, polylogs_package=True, lib_path=None):
    """Start a Wolfram session and return it"""

    if kernel_path is None:
        session = WolframLanguageSession()
    else:
        session = WolframLanguageSession(kernel=kernel_path)
    session.start()

    if polylogs_package:
        if lib_path is not None:
            session.evaluate(wlexpr('$HPLPath = "{}"'.format(lib_path)))
            poly_path = session.evaluate(wl.SetDirectory(lib_path))
            session.evaluate(wlexpr('$PolyLogPath = "{}"'.format(poly_path)))
        session.evaluate(wl.Get('PolyLogTools`'))
    return session


def end_wolfram_session(session):
    """Terminate session"""
    session.terminate()


@timeout(5)
def compute_symbol_wolfram(session, mma_function, symbol_factor=False):
    """For a function given in Mathematica form, we compute its symbol
    Return it as a string type"""

    result = export(session.evaluate(wl.TimeConstrained(wl.PolyLogTools.ComputeSymbol(wlexpr(mma_function),
                                                                                      ConstantEntries=False), 5)))
    result = result.decode('utf-8')
    if symbol_factor:
        result = export(session.evaluate(wl.TimeConstrained(wl.PolyLogTools.SymbolFactor(wlexpr(result)), 5)))
        result = result.decode('utf-8')

    str_result = session.evaluate("ToString[FortranForm[{}]]".format(result))
    new_str_result = str_result.replace('CiTi', 'citi')

    return new_str_result


@timeout(50)
def check_symbol(hyp, tgt, session):
    """Check that the symbol of the hypothesis matches that of the target"""

    mma_hyp = sp.mathematica_code(hyp)
    mma_tgt = sp.mathematica_code(tgt)
    mma_full = '(' + mma_hyp + ') - (' + mma_tgt + ')'
    result = export(session.evaluate(wl.TimeConstrained(wl.PolyLogTools.ComputeSymbol(wlexpr(mma_full),
                                                                                      ConstantEntries=False), 5)))
    str_result = session.evaluate(wl.TimeConstrained("ToString[FortranForm[{}]]".format(result.decode('utf-8')), 5))

    if str_result == '0':
        return True, str_result
    else:
        return False, str_result


def compare_symbol_function(hyp_func, tgt_symb, session):
    """Check that the hypothesis, the function, has the same symbol as the target"""

    mma_hyp_func = sp.mathematica_code(hyp_func)
    mma_tgt = sp.mathematica_code(tgt_symb)

    symb_func = export(session.evaluate(wl.TimeConstrained(wl.PolyLogTools.ComputeSymbol(wlexpr(mma_hyp_func),
                                                                                         ConstantEntries=False), 5)))
    symb_tgt = export(session.evaluate(wlexpr(mma_tgt)))
    str_result = session.evaluate("ToString[FortranForm[SymbolExpand[{}-{}]]]".format(symb_func.decode('utf-8'), symb_tgt.decode('utf-8')))

    if str_result == '0':
        return True, str_result
    else:
        return False, str_result


def split_list(input_list, n):
    """

    Splits a list into n chunks

    :param input_list:
    :param n:
    :return:
    """
    k, m = divmod(len(input_list), n)
    return [input_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def add_symbol_info_temp_file(params, symbol_lines, symbol_lines_mma, lines_prefix, lines_infix, mma_rewrite, per_done):
    """

    Create a temporary file where you add all of the symbol information to the prefix and infix files
    Will also rewrite the symbol files as some of them might need to be discarded

    :param params:
    :param symbol_lines:
    :param symbol_lines_mma:
    :param lines_prefix:
    :param lines_infix:
    :param mma_rewrite:
    :param per_done:
    :return:
    """

    fprefix, new_prefix_path = mkstemp()
    new_prefix_file = fdopen(fprefix, 'a')

    finfix, new_infix_path = mkstemp()
    new_infix_file = fdopen(finfix, 'a')

    fsymb, new_symbol_path = mkstemp()
    new_file_symbol = fdopen(fsymb, 'a')

    if mma_rewrite:
        fsymb_mma, new_symbol_path_mma = mkstemp()
        new_file_symbol_mma = fdopen(fsymb_mma, 'a')
    else:
        new_file_symbol_mma = None
        new_symbol_path_mma = None

    logger.info("Preparing to add {} symbol lines in file {} ".format(len(symbol_lines), new_prefix_path))

    add_symbol_output(params, symbol_lines, symbol_lines_mma, lines_prefix, lines_infix, mma_rewrite,
                      new_file_symbol_mma, new_file_symbol, new_prefix_file, new_infix_file)

    logger.info("Finished batch {} %".format(round(per_done, 2)))

    return new_prefix_path, new_infix_path, new_symbol_path, new_symbol_path_mma


def add_symbol_output(params, symbol_lines, symbol_lines_mma, prefix_lines, infix_lines, mma_rewrite,
                      new_file_symbol_mma, new_file_symbol, new_file_prefix, new_file_infix):
    """

    Takes the relevant symbol information along with the prefix information and combine both into a new file.
    Invalid symbol expressions are discarded along the way and the symbol output file is modified in consequence.

    :param params:
    :param symbol_lines:
    :param symbol_lines_mma:
    :param prefix_lines:
    :param infix_lines:
    :param mma_rewrite:
    :param new_file_symbol_mma:
    :param new_file_symbol:
    :param new_file_prefix:
    :param new_file_infix:
    :return:
    """

    # Necessary to get around pickling issues
    env = CharSPEnvironment(params)
    start_time = time.time()

    for line_num, line_s in enumerate(symbol_lines):
        assert (line_num <= len(prefix_lines))
        # Need to avoid lines that are too long or sympy might not be able to parse in the first place
        if len(line_s) > 10 * env.max_len:
            continue
        new_line = line_s.replace('CiTi', 'citi')
        sympy_expr_line = sp.nsimplify(parse_expr(new_line), tolerance=0.00001, rational=True)
        try:
            prefix1 = env.sympy_to_prefix(sympy_expr_line)
        except UnknownSymPyOperator:
            continue

        # Skip the long inputs
        if len(prefix1) + 2 > 2 * env.max_len:
            continue

        new_file_symbol.write(line_s)
        new_file_symbol.flush()

        if mma_rewrite:
            new_file_symbol_mma.write(symbol_lines_mma[line_num])
            new_file_symbol_mma.flush()

        infix1 = env.prefix_to_infix(prefix1)[1:-1]

        prefix_line = prefix_lines[line_num].split('\t')
        prefix_line[0] = ' '.join(prefix1)
        new_file_prefix.write(f'{prefix_line[0]}\t{prefix_line[1]}')
        new_file_prefix.flush()

        infix_line = infix_lines[line_num].split('\t')
        infix_line[0] = infix1
        new_file_infix.write(f'{infix_line[0]}\t{infix_line[1]}')
        new_file_infix.flush()
        percent_done = line_num / len(prefix_lines) * 100
        if line_num % 1000 == 0:
            time1 = time.time()
            logger.info("Importing symbol --->  {}%".format(str(round(percent_done))))
            print("--- %s eqs/seconds ---" % (1000/(time1 - start_time)))
            start_time = time1


def add_extended_symbol_info(env, filepath_new, filepath_old_pre, filepath_old_in, repet_num, repet_info=None):
    """Read the output of the Mathematica script that contains equivalent
    symbol representations to the one that has been fed as input"""

    new_symbols = open(filepath_new, "r")
    symbol_lines = new_symbols.readlines()

    old_prefix = open(filepath_old_pre, "r")
    prefix_lines = old_prefix.readlines()

    old_infix = open(filepath_old_in, "r")
    infix_lines = old_infix.readlines()

    if repet_info is not None:
        repet_inf = open(repet_info, "r")
        info_lines = repet_inf.readlines()
        assert (len(symbol_lines) == len(info_lines))

    fprefix, abs_path_prefix = mkstemp()
    finfix, abs_path_infix = mkstemp()
    new_prefix_file = fdopen(fprefix, 'a')
    new_infix_file = fdopen(finfix, 'a')

    assert (len(prefix_lines) == len(infix_lines))
    assert (len(symbol_lines) == repet_num * len(prefix_lines))

    num_discarded = 0

    for line_num, line_s in enumerate(prefix_lines):

        # Start by adding the old lines
        prefix_line = line_s.split('\t')
        if len(prefix_line) < 2 or '\x00' in line_s:
            logger.warning("Skipping blank line")
            continue
        try:
            if repet_info is None:
                new_prefix_file.write(f'{prefix_line[0]}\t{prefix_line[1]}')
                new_prefix_file.flush()

                infix_line = infix_lines[line_num].split('\t')
                new_infix_file.write(f'{infix_line[0]}\t{infix_line[1]}')
                new_infix_file.flush()
            else:
                new_prefix_file.write(f'repet0|{prefix_line[0]}\t{prefix_line[1]}')
                new_prefix_file.flush()

                infix_line = infix_lines[line_num].split('\t')
                new_infix_file.write(f'repet0|{infix_line[0]}\t{infix_line[1]}')
                new_infix_file.flush()
        except IndexError:
            logger.error('Index error reached on line of length {}'.format(str(len(prefix_line))))

        # Now add the new contributions

        for repet_number in range(repet_num):
            new_symbol = symbol_lines[line_num*repet_num + repet_number]
            if 'Failed' in new_symbol:
                num_discarded += 1
                continue
            new_line = new_symbol.replace('CiTi', 'citi')
            try:
                sympy_expr_line = sp.nsimplify(parse_expr(new_line), tolerance=0.00001, rational=True)
            except:
                logger.info("Error at parse expr {}".format(new_line))
                logger.info("Line {}".format(line_num))
                num_discarded += 1
                continue

            if repet_info is not None:
                scr_num = info_lines[line_num][0]

            try:
                prefix1 = env.sympy_to_prefix(sympy_expr_line)
            except UnknownSymPyOperator:
                num_discarded += 1
                continue
            try:
                infix1 = env.prefix_to_infix(prefix1)[1:-1]
            except:
                logger.info("Error at expr {}".format(prefix1))
                logger.info("Line {}".format(line_num))
                num_discarded += 1
                continue

            # Skip the long inputs
            if len(prefix1) + 2 > 2 * env.max_len:
                num_discarded += 1
                logger.info("Discarded symbol entry as it is too long --> So far {} discarded".format(num_discarded))
                continue

            new_prefix = ' '.join(prefix1)

            if repet_info is None:
                new_prefix_file.write(f'{new_prefix}\t{prefix_line[1]}')
                new_prefix_file.flush()

                new_infix_file.write(f'{infix1}\t{infix_line[1]}')
                new_infix_file.flush()
            else:
                new_prefix_file.write(f'repet{scr_num}|{new_prefix}\t{prefix_line[1]}')
                new_prefix_file.flush()

                new_infix_file.write(f'repet{scr_num}|{infix1}\t{infix_line[1]}')
                new_infix_file.flush()

        percent_done = line_num / len(prefix_lines) * 100
        if line_num % 5000 == 0:
            logger.info("Importing symbol --->  {}%".format(str(round(percent_done))))

    new_symbols.close()
    old_prefix.close()
    old_infix.close()

    # Copy the file permissions from the old file to the new file
    copymode(filepath_old_pre, abs_path_prefix)
    copymode(filepath_old_in, abs_path_infix)
    # Remove original file
    remove(filepath_old_pre)
    remove(filepath_old_in)
    # Move new file
    move(abs_path_prefix, filepath_old_pre)
    move(abs_path_infix, filepath_old_in)

"""Routines to generate the data relevant for simplification of dilogarithms"""

import os
import random
import glob
from utils_env import scr_null_state, scr_null_state_v2, generate_transformer_starts
import numpy as np
import sympy as sp
from multiprocessing import Pool
from observations.spaces import StringObs
from model.classical_algorithm import canonical_form_expression
import time


def write_transformer_starts(args):
    """
    Generate a dataset that can be fed to a transformer network
    :param args:
    :return:
    """

    worker_id = args[0]
    variable = args[1]
    action_list = args[2]
    num_eqs = args[3]
    save_dir = args[4]
    name_experiment = args[5]
    max_length_simple = args[6]
    max_zero = args[7]
    max_scr = args[8]
    start_time = time.time()
    obs_rep = StringObs(sp.polylog(2, variable), 512, numeral_decomp=True)
    f1 = open(save_dir + name_experiment + str(worker_id) + "_starts_sp.txt", "a")
    f2 = open(save_dir + name_experiment + str(worker_id) + "_starts_prefix.txt", "a")
    skipped = 0

    for eq in range(num_eqs):
        num_terms_simple = random.randint(0, max_length_simple)
        num_zero = random.randint(0, max_zero) if num_terms_simple != 0 else random.randint(1, max_zero)
        total_scr = random.randint(max(1, num_zero), max_scr)
        eq_scr, simple_eq = generate_transformer_starts(num_terms_simple, num_zero, total_scr, 2, 2, variable,
                                                        action_list)

        prefix_scr = ' '.join(obs_rep.sympy_to_prefix(sp.sympify(eq_scr)))
        prefix_simple = ' '.join(obs_rep.sympy_to_prefix(sp.sympify(simple_eq)))

        if len(prefix_scr.split(' ')) > 512:
            skipped += 1
            continue

        f1.write(f'simple{num_terms_simple}zeros{num_zero}scrambles{total_scr}|{eq_scr}\t{simple_eq}\n')
        f1.flush()

        f2.write(f'simple{num_terms_simple}zeros{num_zero}scrambles{total_scr}|{prefix_scr}\t{prefix_simple}\n')
        f2.flush()

        if eq % 1000 == 0 and eq > 0:
            time1 = time.time()
            print("---Worker {} did {} equations".format(worker_id, str(eq+1)))
            print("--- %s eqs/seconds ---" % (1000/(time1 - start_time)))
            start_time = time1
    print('Skipped {} equations'.format(skipped))


def concatenate_files(file_list, new_file_path):
    """
    Concatenation -- can also do it in the command line
    :param file_list:
    :param new_file_path:
    :return:
    """
    with open(new_file_path, 'w') as outfile:
        for fname in file_list:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def generate_null_starts_for_rl(outdir, args, num_eqs, type_gen='v1'):
    """
    Generate random linear combinations that can be simplified down to 0
    Can specify the variable, list of actions, coefficients of rational arguments and number of scrambles
    :param outdir:
    :param args:
    :param num_eqs:
    :param type_gen:
    :return:
    """

    # Unpack the arguments
    variable = args['variable']
    action_list = args['actions']
    max_degree = args['max_degree']
    max_coeff = args['max_coeff']
    num_scr = args['num_scr']
    num_add_zero = args['num_add_zero']

    # Annotate a unique outfile
    outpath = outdir + "/starts.z{}.s{}".format(num_add_zero, num_scr) + ".txt"

    f = open(outpath, "a")
    f2 = open(outpath.replace('.txt', "_mma.txt"), "a")

    print('Generating a dataset with {} total scrambles by adding zero {} times'.format(num_scr, num_add_zero))
    print('Saving the {} equations at {}'.format(num_eqs, outpath))

    for i in range(num_eqs):
        expr_generated = 0
        while expr_generated == 0:

            # Can have different modes for generation
            if type_gen == 'v1':
                expr_generated = scr_null_state(num_scr, num_add_zero, max_degree, max_coeff, variable, action_list)
            elif type_gen == 'v2':
                expr_generated = scr_null_state_v2(num_scr, num_add_zero, max_degree, max_coeff, variable, action_list)
            else:
                raise NameError('Need to choose a valid generator')

        # Write to the data file
        f.write("Example {}: {} scrambles on {} different terms".format(str(i + 1), num_scr, num_add_zero) + '\n')
        f.write('Simple expression : {}'.format(str(0)) + '\n')
        f.write('Scrambled expression : {}'.format(str(expr_generated)) + '\n')
        f.write('\n')

        f2.write("Example {}: {} scrambles on {} different terms".format(str(i + 1), num_scr, num_add_zero) + '\n')
        f2.write('Simple expression : {}'.format(str(sp.mathematica_code(0)) + '\n'))
        f2.write('Scrambled expression : {}'.format(str(sp.mathematica_code(expr_generated))) + '\n')
        f2.write('\n')

        # Keep track of progress
        if i % 50 == 0:
            print("Doing expression {}".format(i))

    f.close()
    f2.close()


def write_to_file(file_path, lines_to_write):
    """Write the lines to the given file path"""

    print('Writing to {} ...'.format(file_path), end=" ")
    f = open(file_path, "w")

    for line in lines_to_write:
        f.write(line)
    f.close()
    print('Done')


def concatenate_files_command(direc, mma=False):
    """
    Faster concatenation
    :param direc:
    :param mma:
    :return:
    """
    allfiles = glob.glob(os.path.join(direc, "*.txt"))
    if not mma:
        files = [file for file in allfiles if 'mma' not in file and 'combined' not in file]
    else:
        files = [file for file in allfiles if 'mma' in file and 'combined' not in file]
    command = "cat "
    for file in files:
        print('Add {} to concatenation list'.format(file))
        command += file + " "

    add_str = "combined.txt" if not mma else "combined_mma.txt"

    command += "> " + direc + add_str
    _ = os.popen(command)
    print('Done')


def split_test_train(alldatadir, percent_test):
    """
    Do an automatic split into a training and a test set
    :param alldatadir:
    :param percent_test:
    :return:
    """
    allfiles = glob.glob(os.path.join(alldatadir, "*.txt"))
    sp_files = [file for file in allfiles if 'mma' not in file]

    for i, file in enumerate(sp_files):
        print('Opening file {}'.format(file), end=" ")
        fsp = open(file, "r")
        speqs = fsp.readlines()

        print("Done")
        sp_separate = [speqs[xmarker:xmarker+4] for xmarker in range(0, len(speqs), 4)]

        num_test = int((len(sp_separate) / 100) * percent_test)
        index_sample = random.sample(range(len(sp_separate)), num_test)

        test_sp = [sp_separate[i] for i in sorted(index_sample)]
        train_sp = [sp_eq for sp_eq in sp_separate if sp_eq not in test_sp]

        assert len(test_sp) + len(train_sp) == len(sp_separate)

        print('Did the split into test/train')

        out_sp_train = file.replace('all_data', 'train_data')
        write_to_file(out_sp_train,  [item for sublist in train_sp for item in sublist])
        out_sp_test = file.replace('all_data', 'test_data')
        write_to_file(out_sp_test, [item for sublist in test_sp for item in sublist])


def save_fit_for_env(datapath, obs_rep):
    """
    Check if the equations being generated are not too long
    :param datapath:
    :param obs_rep:
    :return:
    """
    print('Read Equations from {}'.format(datapath))
    start_f = open(datapath, "r")
    initial_eqs = start_f.readlines()

    print('Converting {} equations to sympy form...'.format(int(len(initial_eqs)/4)), end=" ")
    eq_info = [eqs for eqs in initial_eqs[0::4]]
    eq_info_complex = [(int(eq[eq.find(" scrambles")-1]), int(eq[eq.find(" different terms") - 1])) for eq in eq_info]
    scr_expr = [sp.sympify(eqs[:-1].split(': ')[-1]) for eqs in initial_eqs[2::4]]
    print("Done")

    unique_label = np.unique(np.array(eq_info_complex), axis=0)
    dict_valid_count = {'s'+str(label[0])+'t'+str(label[1]): 0 for label in unique_label}
    dict_total_count = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
    valid_lines = []

    print('"Checking validity ...', end=" ")
    for i, eq in enumerate(scr_expr):
        dict_total_count['s'+str(eq_info_complex[i][0])+'t'+str(eq_info_complex[i][1])] += 1
        valid_len = len(obs_rep.sympy_to_prefix(eq)) < (0.95 * obs_rep.dim)
        if valid_len:
            dict_valid_count['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1
            valid_lines.extend(initial_eqs[i*4:i*4+4])
        if i % 1000 == 0:
            print('"Checking validity ... {} % done'.format(int(100*i/len(scr_expr))))

    print("Done")

    print("Total distribution {}".format(dict_total_count))
    print("Total fit distribution {}".format(dict_valid_count))

    write_to_file(datapath.replace('.txt', '_fit.txt'), valid_lines)


def convert_to_starts(in_path):
    """
    Generate the file for the different episode starts
    Also contains the relevant information on how the data was generated
    :param in_path:
    :return:
    """
    file = open(in_path, 'r')
    lines = file.readlines()

    file_out = open(in_path.replace('.txt', '_rl.txt'), 'w')

    for i, line in enumerate(lines):
        info, funcs = line.split('|')
        num_scr = info.split('scrambles')[-1]
        num_add_zero = info.split('zeros')[-1][0]

        expr_generated, simple_expr = funcs.split('\t')

        file_out.write("Example {}: {} scrambles on {} different terms".format(str(i + 1), num_scr, num_add_zero) + '\n')
        file_out.write('Simple expression : {}'.format(simple_expr))
        file_out.write('Scrambled expression : {}'.format(str(expr_generated)) + '\n')
        file_out.write('\n')


def convert_to_canonical_mp(args):
    """Multiprocessing wrapper"""
    worker_id = args[0]
    path = args[1]
    observ_rep = args[2]
    convert_to_canonical(path+'_{}'.format(str(worker_id)), observ_rep, worker_id)


def convert_to_canonical(path_set, observ_rep, worker_id=0):
    """
    Write the simplified form in canonical form to avoid learning the equivalence
    :param path_set:
    :param observ_rep:
    :param worker_id:
    :return:
    """
    print("Loading equations from {}".format(path_set))
    start_f = open(path_set, "r")
    initial_eqs = start_f.readlines()
    start_time = time.time()

    f_out = open(path_set + '_no_redundancy', "w")
    print("Output equations to {}".format(path_set + '_no_redundancy'))

    split_eqs = [eq.split('\t') for eq in initial_eqs]

    for i, eq in enumerate(split_eqs):
        parsed_simplified = canonical_form_expression(sp.sympify(observ_rep.prefix_to_sympy(eq[1][:-1].split(' '))))[0]
        eq_no_degen = ' '.join(observ_rep.sympy_to_prefix(parsed_simplified))
        f_out.write(f'{eq[0]}\t{eq_no_degen}\n')
        f_out.flush()

        if i % 1000 == 0 and i > 0:
            time1 = time.time()
            print("---Worker {} did {} equations".format(worker_id, str(i+1)))
            print("--- %s eqs/seconds ---" % (1000/(time1 - start_time)))
            start_time = time1


if __name__ == '__main__':

    # Define the observation space
    obs_rep = StringObs(sp.polylog(2, sp.Symbol('x')), 512, numeral_decomp=True)

    # GENERATING DATA FOR TRANSFORMERS ########

    # Define the number of workers and the arguments
    name_exp = 'new_data'
    num_cpu = 4
    x = sp.Symbol('x')
    num_equations = 5000
    actions = ['inversion', 'reflection', 'duplication']
    dir_save = 'example_path'
    max_l_simple = 3
    max_add_zero = 3
    max_scramble = 10

    cpu_id = range(0, num_cpu)
    variables = [x for _ in range(num_cpu)]
    action_lists = [actions for _ in range(num_cpu)]
    num_eqss = [num_equations for _ in range(num_cpu)]
    save_dirs = [dir_save for _ in range(num_cpu)]
    name_exps = [name_exp for _ in range(num_cpu)]
    max_l_simples = [max_l_simple for _ in range(num_cpu)]
    max_add_zeros = [max_add_zero for _ in range(num_cpu)]
    max_scrambles = [max_scramble for _ in range(num_cpu)]

    arguments = list(zip(cpu_id, variables, action_lists, num_eqss, save_dirs, name_exps, max_l_simples, max_add_zeros,
                         max_scrambles))

    # Generate the transformer data
    p = Pool(num_cpu)
    p.map(write_transformer_starts, arguments)
    p.close()
    p.join()

    file_names_sp = [dir_save + name_exp + str(worker_id) + "_starts_sp.txt" for worker_id in range(0, num_cpu)]
    file_names_prefix = [dir_save + name_exp + str(worker_id) + "_starts_prefix.txt" for worker_id in range(0, num_cpu)]

    concatenate_files(file_names_sp, dir_save + name_exp + "_starts_sp.txt")
    concatenate_files(file_names_prefix, dir_save + name_exp + "_starts_prefix.txt")

    for f in file_names_sp+file_names_prefix:
        os.remove(f)

    path_in = 'example_path'
    paths = [path_in for _ in range(num_cpu)]
    obs_reps = [obs_rep for _ in range(num_cpu)]

    # To rewrite a training file in canonical form
    arguments = list(zip(cpu_id, paths, obs_reps))
    p = Pool(30)
    p.map(convert_to_canonical_mp, arguments)
    p.close()
    p.join()
    exit()

    # GENERATING DATA FOR RL #########
    x = sp.Symbol('x')
    actions = ['inversion', 'reflection', 'duplication']
    num_equations = 1000
    dir_save = 'data_rl_v2'
    os.makedirs(dir_save, exist_ok=True)
    arg_len = [[1, 1], [2, 1], [2, 2], [3, 1], [3, 2], [3, 3], [4, 1], [4, 2], [4, 3], [4, 4], [5, 1], [6, 1]]
    for arg_l in arg_len:
        num_scr = arg_l[0]
        num_add_zero = arg_l[1]

        argsf = {'variable': x, 'actions': actions, 'max_degree': 2, 'max_coeff': 2, 'num_scr': num_scr,
                 'num_add_zero': num_add_zero}

        generate_null_starts_for_rl(dir_save, argsf, num_equations, type_gen='v2')
    exit()

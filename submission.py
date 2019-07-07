# Import your files here...
# encoding=UTF-8
from collections import defaultdict
import numpy as np
import math
import copy


def read_file(filename):
    L = []
    with open(filename) as file:
        for line in file:
            for x in line.split('\n'):
                if x != '':
                    # remove the right space of each line
                    L.append(x.rstrip(' '))
    return L


def smoothing_transition(matrix):
    # apply the add-1 smoothing rule
    n = len(matrix)
    m = len(matrix[0])
    matrix_1 = np.zeros((n, m))
    # the state END has no next state
    for i in range(n - 1):
        # state = BEGIN
        sum = np.sum(matrix[i], 0) + n - 1
        for j in range(m):
            # next state can't be BEGIN
            if j != m - 2:
                matrix_1[i][j] = (matrix[i][j] + 1) / sum

    return matrix_1


def smoothing_emission(matrix):
    # apply the add-1 smoothing method to compute emission rate
    n = len(matrix)
    m = len(matrix[0])
    matrix_1 = np.zeros((n, m))
    for i in range(n - 2):
        sum = np.sum(matrix[i], 0) + m
        for j in range(m):
            matrix_1[i][j] = (matrix[i][j] + 1) / sum
    return matrix_1


def absolute_discounting(matrix):
    # apply the absolute discounting method to compute emission rate
    n = len(matrix)
    m = len(matrix[0])
    matrix_1 = np.zeros((n,m))
    for i in range(n-2):
        sum = np.sum(matrix[i],0)
        v = np.count_nonzero(matrix[i])
        Ts = 0
        for j in range(m):
            if matrix[i][j] != 0:
                Ts += matrix[i][j]
        p = 1/(Ts+v)
        for j in range(m):
            if matrix[i][j] != 0:
                matrix_1[i][j] = (matrix[i][j] / sum) - p
            else:
                matrix_1[i][j] = v * p / (m - v)
    return matrix_1


def viterbi_calculation_topk(transition, emission, query, top_k):
    states = len(transition) - 2
    symbols = len(query)
    value_dict = defaultdict(list)
    path_dict = defaultdict(list)
    value = []
    for i in range(symbols):
        if i == 0:
            for j in range(states):
                value_dict['{}{}{}'.format(0, '-', j)].append(
                    math.log(transition[states][j]) + math.log(emission[j][query[i]]))
        else:
            for j in range(states):
                candidate = []
                for k in range(states):
                    for l in range(len(value_dict['{}{}{}'.format(i - 1, '-', 0)])):
                        candidate.append(
                            value_dict['{}{}{}'.format(i - 1, '-', k)][l] + math.log(transition[k][j]) + math.log(
                                emission[j][query[i]]))
                candidate_s = copy.deepcopy(candidate)
                candidate.sort()
                pop_before = 1
                for a in range(min(top_k, len(candidate_s))):
                    pop_value = candidate.pop()
                    value_dict['{}{}{}'.format(i, '-', j)].append(pop_value)
                    if pop_value != pop_before:
                        candidate_index = candidate_s.index(pop_value)
                    else:
                        candidate_index = candidate_s.index(pop_value, candidate_index_before + 1)
                    pop_before = copy.deepcopy(pop_value)
                    candidate_index_before = copy.deepcopy(candidate_index)
                    path_index = candidate_index // len(value_dict['{}{}{}'.format(i - 1, '-', 0)])
                    path_dict['{}{}{}'.format(i, '-', j)].append(path_index)

    value_list = []
    last_path = []
    for i in range(states):
        for j in range(len(value_dict['{}{}{}'.format(symbols - 1, '-', 0)])):
            value_list.append(value_dict['{}{}{}'.format(symbols - 1, '-', i)][j])
    value_list_s = copy.deepcopy(value_list)
    value_list.sort()
    pop_before = 1
    for a in range(top_k):
        pop_value = value_list.pop()
        value.append(pop_value)
        if pop_value != pop_before:
            value_index = value_list_s.index(pop_value)
        else:
            value_index = value_list_s.index(pop_value, candidate_index_before + 1)
        pop_before = copy.deepcopy(pop_value)
        candidate_index_before = copy.deepcopy(candidate_index)
        path_index = value_index // len(value_dict['{}{}{}'.format(symbols - 1, '-', 0)])
        last_path.append(path_index)
    past_path = defaultdict(int)
    path_2 = []
    for i in range(top_k):
        path_1 = []
        a = last_path[i]
        path_1.append(a)
        for j in range(symbols - 1, 0, -1):
            XX = ''
            for b in range(len(path_1)):
                XX = str(XX) + '-' + str(path_1[b])
            a = path_dict['{}{}{}'.format(j, '-', a)][past_path['{}'.format(XX)]]
            past_path['{}'.format(XX)] += 1
            path_1.append(a)
        path_1.reverse()
        path_2.append(path_1)

    return value_dict, path_dict, value, last_path, path_2


def parse_query(query_string, symbol_dict, m):
    # parse query_string with ' ' and elements in ['(',')','/',',','&','-']
    # elements in ['(',')','/',',','&','-'] are seem as a symbol
    # convert them into symbols' index
    # for UNK symbols, denote them as len(symbols)
    # input: 'Holistic Health in Ashmore Suites 2 & 5/151 Cotlew, Ashmore, QLD 4214', symbol_dict, m
    # output: [44211, 20702, 44211, 7801, 41079, 154, 410, 19, 1, 564, 14050, 5, 7801, 5, 25, 42580]
    query_1 = []
    S=['(',')','/',',','&','-']
    for word in query_string.split():  # seperate string into words by bracket
        flag = False
        for special in S:
            if special in word:
                flag = True
                break
        if flag:     # means there are special symbols in the word
            letters = ''
            for letter in word:     # seperating word to letters
                if letter in S:
                    if letters:
                        query_1.append(letters)
                    query_1.append(letter)
                    letters = ''
                else:
                    letters += letter
            if letters:
                query_1.append(letters)
        else:               # means there are no special symbols in the word
            query_1.append(word)
    query_2 = []
    for symbol in query_1:
        if symbol in symbol_dict.keys():        # known symbol
            query_2.append(symbol_dict[symbol][0])
        else:                                   # unknown symbol
            query_2.append(m)
    return query_2

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    return top_k_viterbi(State_File, Symbol_File, Query_File,1)


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    state = read_file(State_File)
    symbol = read_file(Symbol_File)
    query = read_file(Query_File)
    n = int(state[0])
    m = int(symbol[0])
    symbol_dict = defaultdict(list)

    # transition is a matrix that record transition frequency
    # state i transits to state j transition[i][j] times
    # emission is a matrix that record emission frequency
    # state i emits to symbol j emission[i][j] times
    # the last column represent the UNK symbol
    transition = np.zeros((n, n))
    emission = np.zeros((n, m + 1))
    for i in range(n + 1, len(state)):
        # f3 is the frequency of seeing f1 transits/emits to f2
        f1, f2, f3 = state[i].split(' ')
        transition[int(f1)][int(f2)] = int(f3)
    for i in range(m + 1, len(symbol)):
        # f3 is the frequency of seeing f1 transits/emits to f2
        f1, f2, f3 = symbol[i].split(' ')
        emission[int(f1)][int(f2)] = int(f3)

    # these two are stochastic matrix with smoothing
    transition_1 = smoothing_transition(transition)
    emission_1 = smoothing_emission(emission)


    for i in range(1, m + 1):
        symbol_dict[symbol[i]].append(i - 1)

    query_2 = []
    for q in query:
        query_2.append(parse_query(q, symbol_dict, m))

    result_2 = []
    for i in range(len(query_2)):
        query = query_2[i]
        value, path, value_max, last_path, path_2 = viterbi_calculation_topk(transition_1, emission_1, query, k)
        value_x = []
        for j in range(len(value_max)):
            H = value_max[j] + math.log(transition_1[path_2[j][-1]][-1])
            value_x.append(H)
        for x in range(len(value_x)):
            H = [len(transition) - 2] + path_2[x] + [len(transition) - 1] + [value_x[x]]
            result_2.append(H)

    return result_2

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    state = read_file(State_File)
    symbol = read_file(Symbol_File)
    query = read_file(Query_File)
    n = int(state[0])
    m = int(symbol[0])
    symbol_dict = defaultdict(list)

    # transition is a matrix that record transition frequency
    # state i transits to state j transition[i][j] times
    # emission is a matrix that record emission frequency
    # state i emits to symbol j emission[i][j] times
    # the last column represent the UNK symbol
    transition = np.zeros((n, n))
    emission = np.zeros((n, m + 1))
    for i in range(n + 1, len(state)):
        # f3 is the frequency of seeing f1 transits/emits to f2
        f1, f2, f3 = state[i].split(' ')
        transition[int(f1)][int(f2)] = int(f3)
    for i in range(m + 1, len(symbol)):
        # f3 is the frequency of seeing f1 transits/emits to f2
        f1, f2, f3 = symbol[i].split(' ')
        emission[int(f1)][int(f2)] = int(f3)

    # these two are stochastic matrix with smoothing
    transition_1 = smoothing_transition(transition)
    emission_1 = absolute_discounting(emission)


    for i in range(1, m + 1):
        symbol_dict[symbol[i]].append(i - 1)

    query_2 = []
    for q in query:
        query_2.append(parse_query(q, symbol_dict, m))

    result_2 = []
    for i in range(len(query_2)):
        query = query_2[i]
        value, path, value_max, last_path, path_2 = viterbi_calculation_topk(transition_1, emission_1, query, 1)
        value_x = []
        for j in range(len(value_max)):
            H = value_max[j] + math.log(transition_1[path_2[j][-1]][-1])
            value_x.append(H)
        for x in range(len(value_x)):
            H = [len(transition) - 2] + path_2[x] + [len(transition) - 1] + [value_x[x]]
            result_2.append(H)

    return result_2
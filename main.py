# imports
from __future__ import annotations
from typing import Deque, List, Optional
# import string 
# import re
from collections import deque, OrderedDict
# import sys
from copy import deepcopy

from machine import Machine



def REtoNFA(reg_expression):
    # XXX: Using Thomson Subset Construciton for RE to NFA
    # assuming a string of fixed format for operators, etc. in the reg_expression
    # read/write is left to right
    # '+' (union), adjacency means concatenate, '*' (Kleene Closure) 
    # TODO: currently the input symbols may not coincide with those of the operators as current code cannot distinguish. Maybe try format with backslash before operator.
    
    # initializing dimensionality of a transition table
    # let n be the the number of states
    # let k be the number of non-empty symbols
    # The number of distinct symbols for outward transitions from any given state is k + 1, with the addition of null transition (epsilon)

    # finding number of distinct input symbols in regex (excluding operators and parentheses)
    reg_expression = reg_expression.strip()
    reg_expression = '(' + reg_expression + ')'
    excluded_symbols = {'(', ')', '+', '*'}
    input_symbols = [symbol for symbol in set(reg_expression) if symbol not in excluded_symbols]  # remove() would have thrown error if element not found... discard() doesn't.
    input_symbols.sort()
    # print(f"input symbols are {input_symbols}")
    # XXX: print the enum_input_symbols to confirm order of symbols' indices for following transition table (nested list)
    # print(input_symbols)
    k = len(input_symbols)
            
    operators_stack = deque()
    operators_stack.append('(')

    # machines: List[Machine] = []
    machines = deque()

    # CONVERTING RE TO POSTFIX 
    # continuous concatenations of characters will be converted into machines
    # machines will be represented by their respective numerical hashes correspoding to their storage in machines list
    # we are converting into postfix to simplify further nested machine operations (Union, Concatenation, Kleene-Closure)

    postfix_machines = ""
    sub_machine = None

    for c in reg_expression:
        if c in input_symbols:
            if sub_machine is None: 
                sub_machine = Machine(input_symbols = input_symbols, hasNull = True)        
                sub_machine.concat(c)
                # print(f"concatted {c}")
            else:
                # print(f"concatted {c}")
                sub_machine.concat(c)
        else:
            if sub_machine is not None: 
                machines.append(sub_machine)
                postfix_machines += f"{len(machines) - 1} "
            sub_machine = None
            if c == ')':
                while(1):
                    topmost = operators_stack.pop()
                    if topmost == '(':
                        break
                    postfix_machines += topmost + " "
            elif c == '*':
                postfix_machines += c + " "
            else:
                # when it is an operator
                operators_stack.append(c)

    # XXX: then why store it as a string in the first place instead of a list in above code???
    postfix_machines_split = postfix_machines.split()
    # print(f"postfix_machines is {postfix_machines}\npostfix_machines_split is {postfix_machines_split}")

    # i = -1
    # for submachine in machines:
    #     i += 1
    #     print(f"SUBMACHINE {i}\n")
    #     print(submachine.relative_transition_table)
    #     print('\n')

    temp_stack = deque()  # will store machines during postfix parsing done below
    for item in postfix_machines_split:
        match(item):
            case '+':
                second = machines[temp_stack.pop()]
                first = machines[temp_stack.pop()]
                # FIXME: is this working???
                new_machine = first.Union(second)
                machines.append(new_machine)
                temp_stack.append(len(machines) - 1)
                # print(f"for Union item {item} the temp_stack is:\n{temp_stack}")
            case '*':
                first = machines[temp_stack.pop()]
                new_machine = first.Kleene()
                machines.append(new_machine)
                temp_stack.append(len(machines) - 1)
                # print(f"for Kleene item {item} the temp_stack is:\n{temp_stack}")
            case _:
                temp_stack.append(int(item))
                # print(f"for item {item} the temp_stack is:\n{temp_stack}")

    # i = -1
    # for submachine in machines:
    #     i += 1
    #     print(f"POST SUBMACHINE {i}\n")
    #     print(submachine.relative_transition_table)
    #     print('\n')


    # Concatenating all remaining machines in temp_stack
    while(len(temp_stack) > 1):
        second = machines[temp_stack.pop()]
        first = machines[temp_stack.pop()]
        new_machine = first.Concatenate(second)
        machines.append(new_machine)
        temp_stack.append(len(machines) - 1)

    result_machine: Machine = machines[temp_stack.pop()]
    result_machine.Finalize(last_updated = 'relative')

    return  result_machine

class ImproperParamsInput(Exception):
    def __init__(self, message="An Improper value was entered as argument"):
        self.message = message
        super().__init__(self.message)
            
def NFAtoDFA(NFA: Machine) -> Machine:
    # defining a function to take NULL-CLOSURE of a given subset of states of the (null or not null-) NFA
    # The NULL-CLOSURE returns a subset of states, which if unique, is added to a set and queue
    # these subsets will be uniquely stored in a set and labeled (enumerated) 
    # those unique labels will correspond to the states of the resulting DFA
    
    if not NFA.hasNull: raise ImproperParamsInput("Machine object passed to NFAtoDFA() must be an NFA.")
    
    # will be using the absolute transition table of given NFA
    input_symbols = deepcopy(NFA.input_symbols)

    def NullClosure(m: Machine, subset: set) -> set:
        if not m.hasNull: return set()  # returns empty set if machine m has no null transitions
        if subset == set(): return set()  # base case of recursion
        machine_cpy: Machine = deepcopy(m)
        subset_cpy: set = deepcopy(set(subset))
        null_slot_index = machine_cpy.symbols_count - 1
        result_set: set = deepcopy(subset_cpy)  # since each state has an implicit null transition to itself
        # the last column in the transition table is for null transitions
        # recursively find null closures of all targets of null transitions for each state in the input subset (subset_cpy)
        # append their results to the result_set, and finally resturn the result_set
        next_target_subset = set()
        for state in subset_cpy:
            # state corresponds to index of state's entry in transition table
            null_slot = machine_cpy.absolute_transition_table[state][null_slot_index]
            if null_slot == [None]: continue
            # recursion risks going on endlessly if null_slot of a particular state contains index of that state itself, which is technically correct and possible but usually redundant 
            # hence original subset subset_cpy must be subtracted from the next_target_subset
            next_target_subset.update(deepcopy(null_slot))
            next_target_subset = next_target_subset.difference(subset_cpy)
            result_set.update(NullClosure(machine_cpy, next_target_subset))
        return result_set
    
    unique_subsets = []  # these subsets will one-one map to states in the DFA
    DFA_final_states = []
    
    def add_to_unique_subsets(x):
        nonlocal unique_subsets
        unique_subsets.append(tuple(x))
        unique_subsets = list(OrderedDict.fromkeys(unique_subsets))
        return
    
    def check_add_final_state(null_closed_subset: tuple):
        nonlocal NFA, DFA_final_states
        for nfa_state in null_closed_subset:
            if nfa_state in NFA.final_states:
                DFA_final_states.append(unique_subsets.index(null_closed_subset))
                DFA_final_states = list(set(DFA_final_states))
                return
        return 

    state_queue = deque()

    init_dfa_state = tuple(NullClosure(NFA, {0})) 
    state_queue.append(init_dfa_state)  # initializing queue with null closure of initial state of NFA
    add_to_unique_subsets(init_dfa_state)

    DFA_table = []  # dimensions would be len(unique_subsets) x (NFA.symbols_count - 1) x m 

    while(state_queue):
        curr_dfa_state: tuple = state_queue.popleft()
        # print(f"CURRENT DFA STATE IS {curr_dfa_state}")
        input_wise_targets = [[None] for _ in range(NFA.symbols_count - 1)]  # recording aggregate set of target states for the non-null inputs for the current dfa state
        # note that while input_wise_targets is initialized as list of lists, it will end up as a list of tuples toward the end of this loop's iteration 

        for input_symbol_index in range(NFA.symbols_count - 1):
            slot: list = input_wise_targets[input_symbol_index]  # slot holds target states for a particular input symbol... this is done for all the nfa states that compose the current dfa state
            for nfa_state in curr_dfa_state:
                slot[:] += deepcopy(NFA.absolute_transition_table[nfa_state][input_symbol_index])  
            slot[:] = [val for val in slot if val is not None]  # removing None values
            slot[:] = list(set(slot))  # removing duplicates
            # print(f"for input index {input_symbol_index}/{NFA.symbols_count - 1} direct target states are:\n{slot}")

            # Now we have all the input wise target states for the current dfa state
            # overwriting all subsets in input_wise_targets with their null closures
            # storing and queueing all the unique ones
            slot[:] = NullClosure(NFA, slot)
            if tuple(slot) not in unique_subsets:
                add_to_unique_subsets(deepcopy(slot))
                state_queue.append(tuple(deepcopy(slot)))
                check_add_final_state(tuple(deepcopy(slot)))

            # print(f"for input index {input_symbol_index}/{NFA.symbols_count - 1} target states after CLOSURE are:\n{slot}")

        input_wise_targets = [tuple(element) for element in input_wise_targets]  # # locking it to a hashable type tuple in sync with its storage in unique_subsets

        # print(f"\nunique_subsets: {unique_subsets}")
        # print(f"\ninput_wise_targets: {input_wise_targets}\n\n")
        
        input_wise_targets[:] = [[unique_subsets.index(element)] for element in input_wise_targets]
        DFA_table.append(deepcopy(input_wise_targets))

    # print(f"\n\n\nDFA TABLE: \n{DFA_table}")

    DFA = Machine(
        input_symbols = input_symbols, 
        abs_trans_values = DFA_table, 
        final_states = DFA_final_states,
        hasNull = False,
    )
    DFA.Finalize(last_updated = 'absolute')
    
    return DFA

                
                
def MinimizeDFA(DFA: Machine) -> Machine:
    # TODO: sanitize input using logic for DFA check

    cpy_DFA: Machine = deepcopy(DFA)

    def StatesMatch(state1: int, state2: int, prev_hash_table: List[List[int]]):
        nonlocal cpy_DFA
        for symbol_index in range(cpy_DFA.symbols_count):
            state1_trans: Optional[int] = cpy_DFA.absolute_transition_table[state1][symbol_index][0]
            state2_trans: Optional[int] = cpy_DFA.absolute_transition_table[state2][symbol_index][0]
            if (state1_trans is None) ^ (state2_trans is None): 
                return False
            elif state1_trans is None and state2_trans is None: 
                continue
            elif prev_hash_table[state1_trans] != prev_hash_table[state2_trans]:
                return False 
        return True

    def DictReverse(ref_dict: dict):
        result_dict = dict()
        for key, val in ref_dict.items():
            try:
                _ = result_dict[val]
                result_dict[val] += [key]
            except:
                result_dict[val] = [key]
        return result_dict

    prev_equivalence_class = [
        list(set([i for i in range(len(DFA.absolute_transition_table))]) - set(deepcopy(DFA.final_states))),  # non-final states
        deepcopy(DFA.final_states), # final states
    ]

    prev_hash_table = dict()
    i = -1
    for group in prev_equivalence_class:
        i += 1
        for state in group:
            prev_hash_table[state] = i
    
    while(1):
        # print(f"\n\nPREV EQUIVALENCE CLASS IS: {prev_equivalence_class}")
        temp_prev_hash_table = dict()
        new_equivalence_class: List[List[int]] = []
        top_val = -1
        
        for group in prev_equivalence_class:
        # finding equivalence status of stats in the group pair-wise
        # two states are considered equivalent if they have matching transition sets for all inputs respectively
            # print(f"GROUP -- {group} from eq class {prev_equivalence_class}")
            top_val += 1
            group_matches_hash_table = {group[0]: top_val}
            for state in group[1:]:
                # print(f"COVERING STATE {state}")
                covered_states = list(group_matches_hash_table.keys())
                for covered_state in covered_states:
                    if StatesMatch(covered_state, state, prev_hash_table):
                        group_matches_hash_table[state] = group_matches_hash_table[covered_state]
                        # print(f'when covered {covered_states} matched states are {covered_state} and {state}')
                if state in group_matches_hash_table.keys(): continue
                # print(f'when covered {covered_states} Unmatched state {state}')
                top_val += 1
                group_matches_hash_table[state] = top_val

            # print(f"matches hash table is:\n{group_matches_hash_table}")

            temp_prev_hash_table.update(group_matches_hash_table)
            # print(f"temp prev hash table is:\n{temp_prev_hash_table}\n")
            new_hash_table_reversed = DictReverse(group_matches_hash_table)
            new_equivalence_class += [new_group for _, new_group in new_hash_table_reversed.items()]

        if new_equivalence_class == prev_equivalence_class: break
        prev_equivalence_class = new_equivalence_class
        prev_hash_table = temp_prev_hash_table

    minDFA_abs_trans_table: List[List[List[Optional[int]]]] = []
    minDFA_final_states = []

    final_ordered_groups = dict()
    i = -1
    for group in new_equivalence_class:
        minDFA_abs_trans_table.append(deepcopy(cpy_DFA.absolute_transition_table[group[0]]))
        i += 1
        if group[0] in cpy_DFA.final_states: minDFA_final_states.append(i) 
        for state in group:
            final_ordered_groups[state] = i

    for entry in minDFA_abs_trans_table:
        for slot in entry:
            slot[0] = None if slot[0] is None else final_ordered_groups[slot[0]]
          
    minDFA: Machine = Machine(
        input_symbols = cpy_DFA.input_symbols,
        abs_trans_values = minDFA_abs_trans_table,
        final_states = minDFA_final_states,
        hasNull = False,
    )
    minDFA.Finalize(last_updated = 'absolute')
            
    return minDFA
                
                
        

def main():
    # input_string = "Dumb1"  # length is 6... dfa_even should give True
    # if dfa_even(input_string):
    #     print("Wallahi its even it wooorks")
    # else:
    #     print("Well that's odd... hmm")

    # DEFINING TEST RE FOR INPUT
    reg_expression = "(a+b)*bb"  # following notation as covered in TOC course

    # INPUT RE
    # GENERATING NFA USING THOMSON'S CONSTRUCTION
    NFA = REtoNFA(reg_expression)
    print(f"\n\n-----NFA-----\n{NFA}")
    # print(NFA.absolute_transition_table)

    # INPUT NFA
    # GENERATING DFA USING SUBSET CONSTRUCTION 
    DFA = NFAtoDFA(NFA)
    print(f"\n\n-----DFA-----\n{DFA}")

    # INPUT NON-MIMINIMUM DFA
    # MINIMIZING DFA GENERATED
    minDFA = MinimizeDFA(DFA)
    print(f"\n\n-----Minimum DFA-----\n{minDFA}")
    

if __name__ == "__main__":
    main()
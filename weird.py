# imports
from __future__ import annotations
from typing import Deque, List
import string 
import re
from collections import deque, OrderedDict
import sys
from copy import deepcopy

# NOTE: instead of manually listing transitions for cartesian product of all states and input_symbols, I implemented a dict subclass capable of regex initialization
class transitions_dict(dict):
    def __init__(self, val):
        super().__init__(val)
        # NOTE: the below patterns should be used in the keys for this dict descendant's object declaration for conciseness
        # TODO: convert into a collection of patterns and refactor this class' __getitem__ accordingly
        self.__re_pattern1 =  string.ascii_letters + string.digits  # covers alphanumeric characters
    def __getitem__(self, key):
        # key will be a two element tuple (source_state, input_character) of the generic <int, char>
        # the char in key[1] is to be matched with the appropriate __re_pattern. as defined in __init__()
        first = key[0]
        if key[1] in self.__re_pattern1:
            key = (first, self.__re_pattern1)
            # HACK: needs to be generalized for more re_patterns and edge cases
        # else:
        #     raise ValueError("Key to retreive 'transitions' item does not contain intended variety of input character pattern. Fix the input keys you initialized this object instance with.")
        return super().__getitem__(key)
        
class dfaMachine:
    def __init__(self, *, input_symbols, states_labels, init_state, transitions, final_states):
        # TODO: sanitize inputs
        self.input_symbols = input_symbols
        self.states_labels = states_labels
        self.init_state = init_state
        self.transitions = transitions_dict(transitions)
        self.final_states = final_states
        # if init_state == None:
        #     init_state = states_labels[0]
        

# implementation of minimum dfa for even numbers
def RunDFA(s: str, machine: dfaMachine):
    # traversing the DFA machine for given string s
    state = machine.init_state
    for c in s:
        state = machine.transitions[(state, c)]
        print(f"character is {c} and state is {state}")
    print(f"The terminal state is {state}")
    if state in machine.final_states:
        return True
    return False

'''
    state = init_state
    for c in s:
        match state:
            case 0:
                if c in input_symbols:
                    state = 1
                else:
                    break
            case 1:
                if c in input_symbols:
                    state = 0
                else:
                    break
            case _:
                return -1  # let this be the error code
    if state in final_states:
        return 1
    return 0
'''
        

def dfa_even(s: str):
    input_symbols = string.ascii_letters + string.digits
    states_labels = (0, 1)
    init_state = states_labels[0]
    final_states = [states_labels[0]]  # XXX
    alphanumerics = string.ascii_letters + string.digits
    # initializing in the same way as would for dict object
    transitions = [
            ((states_labels[0], alphanumerics), states_labels[1]),
            ((states_labels[1], alphanumerics), states_labels[0]),
    ]

    even_machine = dfaMachine(
        input_symbols = input_symbols, 
        states_labels = states_labels, 
        init_state = init_state, 
        final_states = final_states, 
        transitions = transitions
    )
    return RunDFA(s, machine = even_machine)

def dfa_odd(s: str):
    input_symbols = string.ascii_letters + string.digits
    states_labels = (0, 1)
    init_state = states_labels[0]
    final_states = [states_labels[1]]  # XXX
    alphanumerics = string.ascii_letters + string.digits
    # initializing in the same way as would for dict object
    transitions = [
            ((states_labels[0], alphanumerics), states_labels[1]),
            ((states_labels[1], alphanumerics), states_labels[0]),
    ]

    even_machine = dfaMachine(
        input_symbols = input_symbols, 
        states_labels = states_labels, 
        init_state = init_state, 
        final_states = final_states, 
        transitions = transitions
    )
    return RunDFA(s, machine = even_machine)


class Machine:
    def __init__(self, *, rel_trans_values = None, abs_trans_values = None, input_symbols, hasNull = False):
        # self.input_symbols = None
        # self.len_input_symbols = len(self.input_symbols)
        
        self.input_symbols = input_symbols  # other object parameters' initializations are dependent on this, so current logic only supports input_symbols being const, once initialized.
        self.symbols_count = len(self.input_symbols)
        self.enum_input_symbols = dict()
        for i, symbol in enumerate(self.input_symbols):
            self.enum_input_symbols[symbol] = i
        self.hasNull = hasNull
        if hasNull:
            # assuming the argument input_symbols doesn't also contain null character (epsilon)
            self.symbols_count += 1  
        self.relative_transition_table = [[[None] for _ in range(self.symbols_count)]] if rel_trans_values is None else rel_trans_values 
        self.absolute_transition_table = [[[None] for _ in range(self.symbols_count)]] if abs_trans_values is None else abs_trans_values
        # first entry represents the initial state's transitions
        # relative transition table means:
        #   transitions for a particular state are recorded as number additional forward index increments until target state, for each target state (3rd dim), for each input_symbol (2nd dim)
    # TODO: operation methods only meaningfully work if input symbols if two operand machines are the same, and in same sequence in their respective transition tables. Implement for general case.
    def Union(self, second_machine: Machine) -> Machine:
        '''
        Currently this logic is for two input symbols, say {a, b}
        But when >3 input symbols then we can get expressions like (a + b + c + d) in the RE
        We parse this two at a time like: (a + (b + (c + d))) each time creating an extra initial and final state of null transitions
        This is not clean... TODO: implement it cleanly as (a + b + c + d)
        '''
        
        # let the new machine have an initial state and a final state
        # the initial state has transitions toward the respective intial states of the two operand machines
        # the final states of both operand machines each have transition toward the final state of the new machine
        # initalizing transition table of new machine with initial and final state
        new_relative_transition_table = [[[None] for _ in range(self.symbols_count)] for _ in range(2)]  # recall that self.symbols_count is the number of input symbols
        # ADDING THE FIRST MACHINE TO THE NEW MACHINE
        slot = new_relative_transition_table[-2][self.symbols_count - 1]  # points to initial state of new machine
        slot[:] = [x for x in slot if x is not None] + [1]  # null transition to the initial state of first machine that is to be added to the table
        # using slice assignment to insert first machine's transition table into that of new machine...
        new_relative_transition_table[1:1] = deepcopy(self.relative_transition_table)
        slot = new_relative_transition_table[-2][self.symbols_count - 1]  # points to final state of first machine 
        slot[:] = [x for x in slot if x is not None] + [len(second_machine.relative_transition_table) + 1]  # null transition to new machine's final state
        # ADDING SECOND MACHINE TO THE NEW MACHINE
        slot = new_relative_transition_table[0][self.symbols_count - 1]  # points to initial state of new machine
        machine2_initstate_index = len(new_relative_transition_table) - 1
        slot[:] = [x for x in slot if x is not None] + [machine2_initstate_index]  # null transition to the initial state of second machine that is to be added to the table
        new_relative_transition_table[machine2_initstate_index:machine2_initstate_index] = deepcopy(second_machine.relative_transition_table)
        slot = new_relative_transition_table[-2][self.symbols_count - 1]  # points to final state of second machine
        slot[:] = [x for x in slot if x is not None] + [1]  # transition to final state of new machine
        return Machine(rel_trans_values = new_relative_transition_table, input_symbols = self.input_symbols, hasNull = self.hasNull or second_machine.hasNull)
    
    def Kleene(self) -> Machine:
        new_relative_transition_table = [
            [[None] for _ in range(self.symbols_count)],
            [[None] for _ in range(self.symbols_count)],
        ]
        slot = new_relative_transition_table[-2][self.symbols_count - 1]  # points to initial state of new machine
        slot[:] = [x for x in slot if x is not None] + [1]  # null transition to the initial state of input machine that is to be added to the table
        # Adding current machine to new machine
        new_relative_transition_table[1:1] = deepcopy(self.relative_transition_table)
        slot = new_relative_transition_table[-2][self.symbols_count - 1]  # points to final state of current machine
        new_machine_len = len(new_relative_transition_table)
        slot[:] = [x for x in slot if x is not None] + [1]  # null transition to the final state of new machine
        slot[:] = [x for x in slot if x is not None] + [-(new_machine_len - 3)]  # backward null transition to the initial state of current machine
        # Wrt above statement, note again that transitions are relative by index, and not absolute references
        slot = new_relative_transition_table[0][self.symbols_count - 1]  # points to initial state of new machine
        slot[:] = [x for x in slot if x is not None] + [new_machine_len - 1]  # transition to final state of new machine
        return Machine(rel_trans_values = new_relative_transition_table, input_symbols = self.input_symbols, hasNull = self.hasNull)
    
    def Concatenate(self, second_machine: Machine) -> Machine:
        # adding first machine
        new_relative_transition_table = deepcopy(self.relative_transition_table)
        # combining final state of first machine and initial state of second machine
        common_state = []
        first_final = deepcopy(new_relative_transition_table[-1])
        first_final = [] if first_final == [None] else first_final
        second_init = deepcopy(second_machine.relative_transition_table[0])
        first_final = [] if second_init == [None] else second_init
        for symbol_index in range(self.symbols_count):
            common_state.append(list(set(first_final[symbol_index] + second_init[symbol_index])))
        
        _ = new_relative_transition_table.pop(-1)  # deleting last entry of furst machine in new_relative_transition_table
        new_relative_transition_table.append(common_state)  # adding entry for common state
        new_relative_transition_table += deepcopy(second_machine.relative_transition_table[1:])  # adding second machine except for entry of its initial state (already combined into common_state and added)
        return Machine(rel_trans_values = new_relative_transition_table, input_symbols = self.input_symbols, hasNull = self.hasNull or second_machine.hasNull) 
                    
    def concat(self, c) -> None:
        # This function is for some character c and not machines 
        # Add entry to machine's transition table
        self.relative_transition_table.append([[None] for _ in range(self.symbols_count)])
        
        # print(f"concat operation performed for character {c} with enum as {self.enum_input_symbols[c]}\n btw complete enum is {self.enum_input_symbols}")
        slot = self.relative_transition_table[-2][self.enum_input_symbols[c]]
        slot[:] = [x for x in slot if x is not None] + [1]
        return
    
    def Rel2Abs(self) -> None:
        i = -1
        cpy_rel_table = deepcopy(self.relative_transition_table)
        for state_entry in cpy_rel_table:
            i += 1
            for slot in state_entry:
                if slot == [None]: continue
                slot[:] = [x + i for x in slot]
        self.absolute_transition_table = cpy_rel_table
        return
    
    def Abs2Rel(self) -> None:
        i = -1
        cpy_abs_table = deepcopy(self.absolute_transition_table)
        for state_entry in cpy_abs_table:
            i += 1
            for slot in state_entry:
                if slot == [None]: continue
                slot[:] = [x - i for x in slot]
        self.relative_transition_table = cpy_abs_table
        return
    
    class _ImproperParamsInput(Exception):
        def __init__(self, message="An Improper value was entered as argument for Machine class' method."):
            self.message = message
            super().__init__(self.message)

    def Finalize(self, *, last_updated: str) -> None:
        # both relative and absolute tables are updated to synchronize with the specified last updated table 
        allowed_input = {'relative', 'absolute'}
        if last_updated not in allowed_input: raise self._ImproperParamsInput("Improper value passed for last_updated in Machine.Finalize(). Must be either 'relative' or 'absolute'.")
        if last_updated == 'relative':
            self.Rel2Abs()
        else:
            self.Abs2Rel()
        # # machine must have either relative_transition_table or absolute_transition_table properly filled... need to identify which one.
        # # to do that, compare with the initialization value in __init__()
        # table_init_value = [[[None] for _ in range(self.symbols_count)]]
        #     if not self.relative_transition_table == table_init_value: self.Rel2Abs()
        #     if not self.absolute_transition_table == table_init_value: self.Abs2Rel()
        return

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
    # could also return transition table separately instead of the entire Machine object (which is only internally defined in this function)

    '''
    for c in reg_expression:
        if c is '(':
            stack.append(c)
            count.append(0)
            count_last_index = len(count) - 1
        elif c is ')':    
            # pop stack until topmost '(' and push the terminal states (initial and final of subexpression)
            pass
        elif c in input_symbols:
            # label for new state corresponding to read character c
            curr_state_index += 1
            # this directly hashes to the corresponding index of the state in the transition table
            
            def findPrevState(stack: deque, depth = 0):
                # finding topmost element in stack that is not '(' or ')'
                # this recursion is to skip through possible combos of '((((...' in the stack
                # to avoid edge case at the beginning of the expression where it is skipped past outermost '(', we initialized stack with a state label '0'  
                try:
                    stack[-1 - depth]
                except Exception as e:
                    if e is not IndexError: 
                        print(f"Got the following error while trying to access 'stack': {e}")
                        sys.exit(1)
                if stack[-1 - depth] == '(':
                    return findPrevState(depth + 1)
                else:
                    return stack[-1 - depth]
            prev_state_index = findPrevState(stack)
            
            # UPDATING transitions of parent state in transition table with the new state
            element = transition_table[prev_state_index][enum_input_symbols[c]]  # points to a list object
            # NOTE: element was initialized as [None] by newState() so we need to replace it. Else simply appending if previously replaced. 
            if element == [None]:
                element = [curr_state_index]
            else:
                element.append(curr_state_index)
            newState() 

            # pushing new state to stack
            stack.append(curr_state_index)

        elif c is '+':
            # retrace to the state index directly before the first cluster of '((((...' 
            # ignore equal number of '(' for all ')' since there may be multiple subexpressions within the union's branches as well
            # don't just trace 

            curr_state_index += 1

            def findPrevState(stack: deque, depth = 0):
                topmost = stack[-1 - depth]
                complete_parentheses = 0
                if topmost is '(':
                    complete_parentheses -= 1
                elif topmost is ')':
                    complete_parentheses += 1

                if complete_parentheses < 0 and topmost not in {'(', ')'}:
                    return topmost
                return findPrevState(stack, depth - 1)
            prev_state_index = findPrevState(stack)

            element = transition_table[prev_state_index][enum_input_symbols[]]

    '''

            
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
    def add_to_unique_subsets(x):
        nonlocal unique_subsets
        unique_subsets.append(tuple(x))
        unique_subsets = list(OrderedDict.fromkeys(unique_subsets))
        return
    state_queue = deque()

    init_dfa_state = tuple(NullClosure(NFA, {0})) 
    state_queue.append(init_dfa_state)  # initializing queue with null closure of initial state of NFA
    add_to_unique_subsets(init_dfa_state)

    DFA_table = []  # dimensions would be len(unique_subsets) x (NFA.symbols_count - 1) x m 

    while(state_queue):
        curr_dfa_state: tuple = state_queue.popleft()
        print(f"CURRENT DFA STATE IS {curr_dfa_state}")
        input_wise_targets = [[None] for _ in range(NFA.symbols_count - 1)]  # recording aggregate set of target states for the non-null inputs for the current dfa state
        # note that while input_wise_targets is initialized as list of lists, it will end up as a list of tuples toward the end of this loop's iteration 

        for input_symbol_index in range(NFA.symbols_count - 1):
            slot: list = input_wise_targets[input_symbol_index]  # slot holds target states for a particular input symbol... this is done for all the nfa states that compose the current dfa state
            for nfa_state in curr_dfa_state:
                slot[:] += deepcopy(NFA.absolute_transition_table[nfa_state][input_symbol_index])  
            slot[:] = [val for val in slot if val is not None]  # removing None values
            slot[:] = list(set(slot))  # removing duplicates
            print(f"for input index {input_symbol_index}/{NFA.symbols_count - 1} direct target states are:\n{slot}")

            # Now we have all the input wise target states for the current dfa state
            # overwriting all subsets in input_wise_targets with their null closures
            # storing and queueing all the unique ones
            slot[:] = NullClosure(NFA, slot)
            if tuple(slot) not in unique_subsets:
                add_to_unique_subsets(deepcopy(slot))
                state_queue.append(tuple(deepcopy(slot)))

            print(f"for input index {input_symbol_index}/{NFA.symbols_count - 1} target states after CLOSURE are:\n{slot}")

        input_wise_targets = [tuple(element) for element in input_wise_targets]  # # locking it to a hashable type tuple in sync with its storage in unique_subsets

        print(f"\nunique_subsets: {unique_subsets}")
        print(f"\ninput_wise_targets: {input_wise_targets}\n\n")
        
        input_wise_targets[:] = [[unique_subsets.index(element)] for element in input_wise_targets]
        DFA_table.append(deepcopy(input_wise_targets))

    print(f"\n\n\nDFA TABLE: \n{DFA_table}")

    DFA = Machine(input_symbols = input_symbols, abs_trans_values = DFA_table, hasNull = False)
    DFA.Finalize(last_updated = 'absolute')
    
    return DFA

                
                
                
                
                
        

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
    print(NFA.relative_transition_table)
    print(NFA.absolute_transition_table)


    # INPUT NFA
    # GENERATING DFA USING SUBSET CONSTRUCTION 
    DFA = NFAtoDFA(NFA)
    print(DFA)

    # INPUT NON-MIMINIMUM DFA
    # MINIMIZING DFA GENERATED
    

if __name__ == "__main__":
    main()
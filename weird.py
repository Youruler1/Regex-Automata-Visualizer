# imports
from typing import Deque, List
import string 
import re
from collections import deque
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
    input_symbols = list(set(reg_expression).discard(['(', ')', '+', '*']))  # remove() would have thrown error if element not found... discard() doesn't.
    enum_input_symbols = dict()
    for i, symbol in enumerate(input_symbols):
        enum_input_symbols[symbol] = i
    # XXX: print the enum_input_symbols to confirm order of symbols' indices for following transition table (nested list)
    # print(input_symbols)
    k = len(input_symbols)

    # finding number of states 'n'
    '''
    each read input_symbol in reg_expression is addition of 1 state
    each read '+' (concatenation) is addition of 2 states if on initial subexpression else 1 state
    each read '*' (Kleene Closure) is also addition of 2 states if on initial subexpression else 1 state
    each read 
    '''
    
    # transition_table = []
    # newState = lambda init_val = None: transition_table.append([init_val] * (k + 1))
    # stack = deque()

    # # the stack here accounts for state labels like 0,1,2, etc. and corresponding parentheses in the RE for proper sequence in constructions of subexpressions 

    # curr_state_index = 0  # initial state index
    # newState()
    # stack.append(0)
    # count = []
    # count_last_index = None

    class Machine:
        def __init__(self, *, trans_values = None, input_symbols = None):
            # self.input_symbols = None
            # self.len_input_symbols = len(self.input_symbols)
            
            self.relative_transition_table = trans_values
            self.input_symbols = input_symbols
            self.symbols_count = len(self.input_symbols)
            self.enum_input_symbols = dict()
            for i, symbol in enumerate(self.input_symbols):
                self.enum_input_symbols[symbol] = i
            # first entry represents the initial state's transitions
            # relative transition table means:
            #   transitions for a particular state are recorded as number additional forward index increments until target state, for each target state (3rd dim), for each input_symbol (2nd dim)

        # TODO: operation methods only meaningfully work if input symbols if two operand machines are the same, and in same sequence in their respective transition tables. Implement for general case.

        def Union(self, second_machine: Machine) -> Machine:
            # let the new machine have an initial state and a final state
            # the initial state has transitions toward the respective intial states of the two operand machines
            # the final states of both operand machines each have transition toward the final state of the new machine

            # initalizing transition table of new machine with initial and final state
            new_relative_transition_table = [[[None] for _ in range(self.symbols_count + 1)] for _ in range(2)]  # recall that self.symbols_count is the number of input symbols

            # ADDING THE FIRST MACHINE TO THE NEW MACHINE
            slot = new_relative_transition_table[-2][self.symbols_count]  # points to initial state of new machine
            slot[:] = [x for x in slot if x is not None] + [1]  # null transition to the initial state of first machine that is to be added to the table
            # using slice assignment to insert first machine's transition table into that of new machine...
            new_relative_transition_table[1:1] = deepcopy(self.relative_transition_table)
            slot = new_relative_transition_table[-2][self.symbols_count]  # points to final state of first machine 
            slot[:] = [x for x in slot if x is not None] + [1]  # null transition to new machine's final state

            # ADDING SECOND MACHINE TO THE NEW MACHINE
            slot = new_relative_transition_table[0][self.symbols_count]  # points to initial state of new machine
            machine2_initstate_index = len(new_relative_transition_table) - 1
            slot[:] = [x for x in slot if x is not None] + [machine2_initstate_index]  # null transition to the initial state of second machine that is to be added to the table
            new_relative_transition_table[machine2_initstate_index:machine2_initstate_index] = deepcopy(second_machine.relative_transition_table)
            slot = new_relative_transition_table[-2][self.symbols_count]  # points to final state of second machine
            slot[:] = [x for x in slot if x is not None] + [1]  # transition to final state of new machine

            return Machine(trans_values = new_relative_transition_table)
        
        def Kleene(self) -> Machine:
            new_relative_transition_table = [
                [[None] for _ in range(self.symbols_count + 1)],
                [[None] for _ in range(self.symbols_count + 1)],
            ]
            slot = new_relative_transition_table[-2][self.symbols_count]  # points to initial state of new machine
            slot[:] = [x for x in slot if x is not None] + [1]  # null transition to the initial state of input machine that is to be added to the table

            # Adding current machine to new machine
            new_relative_transition_table[1:1] = deepcopy(self.relative_transition_table)

            slot = new_relative_transition_table[-2][self.symbols_count]  # points to final state of current machine
            new_machine_len = len(new_relative_transition_table)
            slot[:] = [x for x in slot if x is not None] + [new_machine_len]  # null transition to the final state of new machine
            slot[:] = [x for x in slot if x is not None] + [-(new_machine_len - 3)]  # backward null transition to the initial state of current machine
            # Wrt above statement, note again that transitions are relative by index, and not absolute references

            slot = new_relative_transition_table[0][self.symbols_count]  # points to initial state of new machine
            slot[:] = [x for x in slot if x is not None] + [new_machine_len - 1]  # transition to final state of new machine

            return Machine(new_relative_transition_table)
        
        def Concatenate(self, second_machine: Machine) -> Machine:
            # adding first machine
            new_relative_transition_table = deepcopy(self.relative_transition_table)
            
            slot = new_relative_transition_table[-1][self.symbols_count]  # points to final state of first machine
            slot[:] = [x for x in slot if x is not None] + [1]  # null transition to initial state of second machine

            # adding second machine
            new_relative_transition_table = new_relative_transition_table + deepcopy(second_machine.relative_transition_table)

            return Machine(trans_values = new_relative_transition_table) 
                        
        def concat(self, c) -> None:
            # This function is for some character c and not machines 

            # Add entry to machine's transition table
            self.relative_transition_table.append([[None] for _ in range(self.symbols_count + 1)])

            # edge-case: if table previously empty, don't need to add transition
            if len(self.relative_transition_table) == 1:
                return
            
            slot = self.relative_transition_table[-2][self.enum_input_symbols[c]]
            slot = [x for x in slot if x is not None] + [1]
            return
            
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
                sub_machine = Machine(input_symbols = input_symbols)        
            else:
                sub_machine.concat(c)
        else:
            if sub_machine is not None: 
                machines.append(sub_machine)
                postfix_machines += f"{len(machines) - 1} "
            sub_machine = None
            if c is ')':
                while(1):
                    topmost = operators_stack.pop()
                    postfix_machines += topmost + " "
                    if topmost is '(':
                        break
            elif c in input_symbols:
                postfix_machines += c + " "
            else:
                # when it is an operator
                operators_stack.append(c)

    # XXX: then why store it as a string in the first place instead of a list in above code???
    postfix_machines_split = " ".split(postfix_machines)

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
            case '*':
                first = machines[temp_stack.pop()]
                new_machine = first.Kleene()
                machines.append(new_machine)
                temp_stack.append(len(machines) - 1)
            case _:
                temp_stack.append(int(item))


    # Concatenating all remaining machines in temp_stack
    while(len(temp_stack) > 1):
        second = machines[temp_stack.pop()]
        first = machines[temp_stack.pop()]
        new_machine = first.Concatenate(second)
        machines.append(new_machine)
        temp_stack.append(len(machines) - 1)

    return  temp_stack.pop()
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

            
        

def main():
    input_string = "Dumb1"  # length is 6... dfa_even should give True
    if dfa_even(input_string):
        print("Wallahi its even it wooorks")
    else:
        print("Well that's odd... hmm")

    # DEFINING TEST RE FOR INPUT
    reg_expression = "(a+b)*bb"  # following notation as covered in TOC course

    # INPUT RE
    # GENERATING NFA USING THOMSON'S CONSTRUCTION


    # INPUT NFA
    # GENERATING DFA USING SUBSET CONSTRUCTION 


    # INPUT NON-MIMINIMUM DFA
    # MINIMIZING DFA GENERATED
    

if __name__ == "__main__":
    main()
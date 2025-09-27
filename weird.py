# imports
import typing
import string 
import re

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


def main():
    input_string = "Dumb1"  # length is 6... dfa_even should give True
    if dfa_even(input_string):
        print("Wallahi its even it wooorks")
    else:
        print("Well that's odd... hmm")

if __name__ == "__main__":
    main()
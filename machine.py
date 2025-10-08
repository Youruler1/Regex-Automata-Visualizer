from __future__ import annotations
from typing import Deque, List, Optional
# import string 
# import re
from collections import deque, OrderedDict
# import sys
from copy import deepcopy

class Machine:
    def __init__(self, *, rel_trans_values = None, abs_trans_values = None, input_symbols, final_states = None, hasNull = False):
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
        self.final_states = final_states
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
        return Machine(
            rel_trans_values = new_relative_transition_table, 
            input_symbols = self.input_symbols, 
            final_states = [len(new_relative_transition_table) - 1],
            hasNull = self.hasNull or second_machine.hasNull
        )
    
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
        return Machine(
            rel_trans_values = new_relative_transition_table, 
            input_symbols = self.input_symbols, 
            final_states = [len(new_relative_transition_table) - 1],
            hasNull = self.hasNull
        )
    
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
        return Machine(
            rel_trans_values = new_relative_transition_table, 
            input_symbols = self.input_symbols, 
            final_states = [len(new_relative_transition_table) - 1],
            hasNull = self.hasNull or second_machine.hasNull,
        ) 
                    
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
    
    def __str__(self):
        return f"\nInput Symbols: {self.input_symbols}\nColumn index mappings of input symbols: {self.enum_input_symbols}\nNull Transitions: {self.hasNull}\nRel-Trans-Table: {self.relative_transition_table}\nAbs-Trans_Table: {self.absolute_transition_table}\nFinal States: {self.final_states}"

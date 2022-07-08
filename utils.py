import numpy as np
from cfg import *

translator: dict = {
     0: '0',
     1: '1',
     2: '2',
     3: '3',
     4: '4',
     5: '5',
     6: '6',
     7: '7',
     8: '8',
     9: '9',
     10: 'A',
     11: 'B',
     12: 'C',
     13: 'D',
     14: 'E',
     15: 'F',
     16: 'G',
     17: 'H',
     18: 'I',
     19: 'J',
     20: 'K',
     21: 'L',
     22: 'M',
     23: 'N',
     24: 'O',
     25: 'P',
     26: 'Q',
     27: 'R',
     28: 'S',
     29: 'T',
     30: 'U',
     31: 'V',
     32: 'W',
     33: 'X',
     34: 'Y',
     35: 'Z',
     36: 'a',
     37: 'b',
     38: 'd',
     39: 'e',
     40: 'f',
     41: 'g',
     42: 'h',
     43: 'n',
     44: 'q',
     45: 'r',
     46: 't',
     47: ' '}

retranslator: dict = {v:k for k,v in translator.items()}

def text_to_input(text: str):
    text = list(map(lambda x: retranslator.get(x,retranslator[x.upper()]), text) )
    return np.array(text).reshape(-1,1)


class InputTable:
    """ 
    Converts text into array of inputs
    """
    @staticmethod
    def get_splits(line: list) -> list:
        # add one as space
        lens = [len(x)+1 for x in line]
        
        group: int = 0
        splits: list = []
        cumsum: int = 0
        for x, word in zip(lens,line):
            cumsum += x
            
            if '\n' in word:
                splits.append(group)
                group += 1
                cumsum = 0
                continue
            
            if cumsum > MAX_CHARS_PER_LINE:
                group += 1
                cumsum = x
                
            splits.append(group)
        
        return splits
    
    @staticmethod
    def build_table(splits: list, line: list) -> list:
        table: list = []
        for split in set(splits):
            logic_map = [x == split for x in splits]
            row = ' '.join(np.array(line)[logic_map])
            row += ' '*(MAX_CHARS_PER_LINE - len(row))
            table.append(row)
            
        return table
        
        
    def __init__(self, line: str):
        self._line = line.split(' ')
        self._splits = self.get_splits(self._line)
        self._line = [x.replace('\n','') for x in self._line]
        self._table = self.build_table(self._splits, self._line)
        
    
    @property
    def table(self) -> list:
        """
        Returns list of arrays
        """
        return [text_to_input(text) for text in self._table]
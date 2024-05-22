#!/usr/bin/env python

# generates a de Bruijn sequence using the prefer high method
# Robert Howie, April 2013

import argparse

class de_bruijn_sequence:
    """Creates a De Bruijn sequence for alphabet size alphabetSize and subsequence length subsequenceLength generated using the prefer high method"""
    def __init__(self, alphabetSize, subsequenceLength):
        self.sequenceLength = alphabetSize**subsequenceLength #calculate length
        de_bruijn_sequence = [] #create list
        [de_bruijn_sequence.append(0) for x in range(subsequenceLength)] #append starting zeroes
        subsequencesUsed = [[0]*subsequenceLength] #add starting zeroes subsequence to the list of subsequences present
        for i in range(self.sequenceLength)[subsequenceLength:]: #for the rest of the sequence
            nextElementNotFound = True #decrementing checker loop for the prefer high method ends once the correct next character is found
            testElement = alphabetSize-1 #try the highest number first, prefer high
            while nextElementNotFound:
                if (de_bruijn_sequence[i-(subsequenceLength-1):i]+[testElement] not in subsequencesUsed): #if the number works
                    de_bruijn_sequence.append(testElement) #append it to the sequence
                    subsequencesUsed.append(de_bruijn_sequence[-subsequenceLength:]) #then add the new subsequence to the list of subsequences already present
                    #print(subsequencesUsed)
                    nextElementNotFound = False #end searcher loop
                else:
                    testElement = testElement-1 #if not found, try the next highest
            #end while
        #end for
        self.de_bruijn_sequence = de_bruijn_sequence
        
    #end __init__
#end class deBruijn

def main():
    #create argument parser
    parser = argparse.ArgumentParser(description="Creates De Bruijn sequences using the prefer high method")
    #positional argument for alphabet size
    parser.add_argument("k", type=int, help="alphabet size")
    #positional argument for subsequence length
    parser.add_argument("n", type=int, help="subsequence length")
    #parse arguments
    arguments = parser.parse_args()
    
    k = arguments.k #alphabet size
    n = arguments.n #subsequence length
    
    particularDeBruijnSequence = de_bruijn_sequence(k, n)
    print("Alphabet Size k = {0}\nSubsequence length n = {1}\nDe Bruijn Sequence Length: {2}\nDe Bruijn Sequence: \n\n".format(k, n, particularDeBruijnSequence.sequenceLength)+"".join(map(str, particularDeBruijnSequence.de_bruijn_sequence))+"\n")

#end main

if __name__ == "__main__":
    main()
#end if  __name__ == "__main__"

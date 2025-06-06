#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  30 23:55:37 2024

@author: ckri0
"""

import nltk
from nltk import CFG
import gensim

# Word2Vec
# location of Word2Vec model
model_path = '/GoogleNews-vectors-negative300.bin'

# load model
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# download model if not locally available
#model = gensim.downloader.load('word2vec-google-news-300')

# define the grammar
grammar = CFG.fromstring("""

    S -> NP VP

    NP -> D N | D N PP | N | N PP | PRON
    VP -> V NP | V NP PP | V PP
    PP -> P NP | P NP PP
    
    D -> 'the' | 'a' | 'his'
    N -> 'sandwich'  | 'fork' | 'aisles' | 'store' | 'problem' | 'teacher' | 'picture' | 'girl' | 'park' | 'boy' | 'dog' | 'stick' | 'ball'
    N -> 'linguistics' | 'corner' | 'book' | 'man'| 'music' | 'room' | 'doctor' | 'patients' | 'hallway' | 'horse' | 'child' | 'knight'  
    N -> 'journalist' | 'author' | 'mother' | 'bottle' | 'fridge' | 'pizza' | 'anchovies' | 'friends' | 'rabies' | 'beer' | 'spice'
    N -> 'movie' | 'airplane' | 'game' | 'seattle' | 'people' | 'milk' | 'idaho' | 'toolbox' | 'hammer' | 'statue' | 'research'
    P -> 'with' | 'in' | 'of' | 'from' | 'about' | 'on' | 'to'
    V -> 'ate' | 'peruse' | 'perused' | 'discussed' | 'drew' | 'hit' | 'heard' | 'talked' | 'admired' | 'gave' | 'interviewed' 
    V -> 'nursed' | 'beat' | 'ate' | 'drank' | 'watched' | 'studied' | 'drank' | 'opened'
    PRON -> 'she' | 'we' | 'he' | 'i'

""")

# sentences with obvious parses
obvious_test_sentences_dict = {"I ate pizza with anchovies" : "PP attaches to NP",
                  "She ate the sandwich with a fork" : "PP attaches to VP",
                  "The boy hit the dog with the stick" : "PP attaches to VP",
                  "The boy hit the dog with rabies" : "PP attaches to NP",
                  "The doctor discussed his research in the hallway" : "PP attaches to VP",
                  "The child admired the statue of the knight on the horse" : "PP attaches to NP",
                  "The mother nursed the child with the bottle in the fridge" : "PP attaches to NP",
                  "The man drank the beer with spice" : "PP attaches to NP",
                  "I watched the movie on the airplane" : "PP attaches to VP",
                  "She drank the milk from Idaho" : "PP attaches to NP",
                  }
# sentences that are more ambiguous
not_obvious_test_sentences_dict = {"He beat the man with the ball" : "PP attaches to NP",
                  "We peruse the aisles in the store" : "PP attaches to NP",
                  "She discussed the problem with the teacher" : "PP attaches to VP",
                  "She heard the music from the room" : "PP attaches to VP",
                  "He drew a picture of the girl in the park" : "PP attaches to VP",
                  "The journalist interviewed the author of the book on linguistics" : "PP attaches to NP",
                  "The man gave the book on linguistics to the boy in the corner" : "PP attaches to NP",
                  "I watched the game in Seattle" : "PP attaches to NP",
                  "The man studied the people in Seattle" : "PP attaches to NP",
                  "He opened the toolbox with a hammer" : "PP attaches to VP"
                  }

# combined obvious and not obvious sentences
combined_test_sentences_dict = obvious_test_sentences_dict | not_obvious_test_sentences_dict

# all vp attachment senteces
vp_test_sentences_dict = {key: value for key, value in combined_test_sentences_dict.items() if value == "PP attaches to VP"}

# all np attachment sentences
np_test_sentences_dict = {key: value for key, value in combined_test_sentences_dict.items() if value == "PP attaches to NP"}

# class for parse object
class ParseResult:
    def __init__(self, bracketed_string, pp_attachment, words_to_compare, similarity_score):
        self.bracketed_string = bracketed_string
        self.pp_attachment = pp_attachment
        self.words_to_compare = words_to_compare
        self.similarity_score = similarity_score

    def __str__(self):
        return f"{self.bracketed_string}, {self.pp_attachment}, Similarity Score for {self.words_to_compare}: {self.similarity_score}"

# create a parser
parser = nltk.ChartParser(grammar)

    
# convert the tree to a bracketed string with labels for processing
def tree_to_bracketed(tree):
    if isinstance(tree, nltk.Tree):
        return f"[{tree.label()} {' '.join(tree_to_bracketed(child) for child in tree)}]"
    else:  # Terminal nodes (words)
        return tree
    
# returns whether a PP attaches to the VP or NP
def find_pp_attachment(parse_tree):
    # recursive function to traverse the tree
    def traverse_tree(tree, parent=None):
        # if the current node is a PP, check its parent
        if tree.label() == 'PP':
            if parent:
                if parent.label() == 'VP':
                    return 'PP attaches to VP'
                elif parent.label() == 'NP':
                    return 'PP attaches to NP'
        
        # recurse through subtrees
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                result = traverse_tree(subtree, parent=tree)
                if result:
                    return result
        
        # if no PP attachment found
        return None
    
    # call the recursive function starting with the root of the parse tree
    return traverse_tree(parse_tree)

# extracts the verb from a parse
def extract_verb(parse_tree):
    # traverse the tree and look for verb nodes
    for subtree in parse_tree.subtrees():
        if subtree.label() == 'V':
            # extract the verb 
            verb = (subtree.leaves()[0])
    return verb

# extracts the noun from the final PP
def extract_pp_noun(parse_tree):
    # initialize blank list to store all PPs
    candidate_pps = []
    # traverse the tree and look for PPs
    for subtree in parse_tree.subtrees():
        if subtree.label() == 'PP':
            candidate_pps.append(subtree)
    
    # check if there is at least one PP
    if not candidate_pps:
        return None  # Return None if no PP found
    
    # get the final PP (last PP in the list)
    final_pp = candidate_pps[-1]
    
    # look for NP in the final PP
    np_subtree = None
    for child in final_pp:
        if child.label() == 'NP':
            np_subtree = child
            break
    
    # check if an NP was found
    if np_subtree is None:
        return None  
    
    # extract the noun from the NP
    for np in np_subtree.subtrees():
        if np.label() == 'N':  
            noun = np.leaves()[0]
            return noun
    
    # if no noun is found
    return None

# extracts the noun from the NP that is part of the VP (object of verb, not subject)
def extract_np_noun(parse_tree):
    # recursive function to traverse the tree
    def traverse_tree(tree, parent=None):
        # check if this node is NP
        if tree.label() == 'NP':  
            # check if the parent is a VP (this NP is the object of the verb)
            if parent and parent.label() == 'VP':
                # extract the noun from the NP
                for np_subtree in tree.subtrees():
                    if np_subtree.label() == 'N':
                        noun = np_subtree.leaves()[0] 
                        return noun 
        
        # recurse through subtrees
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                result = traverse_tree(subtree, parent=tree)  
                if result:  
                    return result
        # if no noun is found
        return None

    # Start traversal from the root of the tree
    return traverse_tree(parse_tree)

def ranked_parse(sentence):
    # clean input for processing
    sentence = sentence.lower().split()
    # initialize blank list of parses
    results = []
    # loop over the parse trees and call find_pp_attachment
    for tree in parser.parse(sentence):
        bracketed_string = tree_to_bracketed(tree)
        pp_attachment = find_pp_attachment(tree)
        words_to_compare = None
        similarity_score = None
        
        if pp_attachment == "PP attaches to VP":
            # extract verb and noun from PP
            verb = extract_verb(tree)
            pp_noun = extract_pp_noun(tree)
            # compute similarity
            if verb and pp_noun:
                words_to_compare = [verb, pp_noun]
                #print(f"Comparing {verb} with {pp_noun}") #uncomment for debugging
                similarity_score = model.similarity(verb, pp_noun)
            else:
                similarity_score = None
        elif pp_attachment == "PP attaches to NP":
            # extract noun from NP and noun from PP
            np_noun = extract_np_noun(tree)
            pp_noun = extract_pp_noun(tree)
            
            # compute similarity
            if np_noun and pp_noun:
                words_to_compare = [np_noun, pp_noun]
                #print(f"Comparing {np_noun} with {pp_noun}") #uncomment for debugging
                similarity_score = model.similarity(np_noun, pp_noun)
            else:
                similarity_score = None
        
        # store result as ParseResult obj
        if words_to_compare is not None and similarity_score is not None:
            result = ParseResult(bracketed_string, pp_attachment, words_to_compare, similarity_score)
            results.append(result)
    
    # sort results based on similarity score (highest first)
    sorted_results = sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    # print all parses ranked by similarity score
    for result in sorted_results:
        print(result)
    
    # newline for legibility
    print()
    
    highest_similarity = sorted_results[0].similarity_score
    
    # even if a sentence has more than two parses, there are only two sets of similarity scores. 
    # algorithm does not properly handle multiple PP attachments, so this guarantees that the second similarity score is selected
    second_similarity = sorted_results[-1].similarity_score
    
    # difference in similarity scores associated with the two attachment points for the (final) PP
    margin = highest_similarity - second_similarity
    
    # return tuple of highest scoring parse and similarity score margin
    return (sorted_results[0], margin)

# evaluate a dictionary of test sentences (keys) against labels of correct PP attachment (values)
def evaluate(test_sentences_dict):
    reported_answers = []
    answer_key = list(test_sentences_dict.values())
    correct = 0
    incorrect = 0
    for sentence in list(test_sentences_dict.keys()):
        reported_answers.append(ranked_parse(sentence))
    for index, (reported_answer, true_answer) in enumerate(zip(reported_answers,answer_key)):
        if reported_answer[0].pp_attachment == true_answer:
            print(f"Index {index}: Correct! Reported: {reported_answer[0].pp_attachment}, Actual: {true_answer}, Margin: {reported_answer[1]}")
            correct += 1
        else:
            print(f"Index {index}: Incorrect! Reported: {reported_answer[0].pp_attachment}, Actual: {true_answer}, Margin: {reported_answer[1]}")
            incorrect += 1
    accuracy = correct/len(answer_key)
    print(f"Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy}")
    
# main function for user input of sentences
def main():
    # Infinite loop for user input
    while True:
        # Get input from the user
        sentence = input("Enter a sentence (or type 'exit' to quit): ").lower().split()
        
        # Exit condition
        if "exit" in sentence:
            print("Exiting the parser. Goodbye!")
            break
    
        # Parse the sentence and output the bracketed string
        try:
            ranked_parse(sentence)
        except ValueError as e:
            print(f"Error parsing sentence: {e}")
            
# evaluation using predefined test dictionaries    
evaluate(obvious_test_sentences_dict)
evaluate(not_obvious_test_sentences_dict)
evaluate(combined_test_sentences_dict)
evaluate(vp_test_sentences_dict)
evaluate(np_test_sentences_dict)

#error analysis
print(model.most_similar('airplane'))
print(model.most_similar('movie'))

# algorithm does not predict I watched (the movie) on the airplane
model.similarity('watched', 'airplane') # 0.06960811

# algorithm predicts I watched (the movie on the airplane)
model.similarity('movie', 'airplane') # 0.19818681

# similarity between V and N of NP (object of verb) 
# not considered by the algorithm
model.similarity('movie', 'watched') # 0.16539595


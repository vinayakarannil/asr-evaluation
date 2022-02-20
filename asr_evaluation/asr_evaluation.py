# Copyright 2017-2018 Ben Lambert

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Primary code for computing word error rate and other metrics from ASR output.
"""
from __future__ import division

from functools import reduce
from collections import defaultdict


from termcolor import colored

# Some defaults
print_instances_p = False
print_errors_p = False
files_head_ids = False
files_tail_ids = False
confusions = False
min_count = 0
wer_vs_length_p = True

# For keeping track of the total number of tokens, errors, and matches
ref_token_count = 0
error_count = 0
match_count = 0
counter = 0
sent_error_count = 0

# For keeping track of word error rates by sentence length
# this is so we can see if performance is better/worse for longer
# and/or shorter sentences
lengths = []
error_rates = []
wer_bins = defaultdict(list)
wer_vs_length = defaultdict(list)
# Tables for keeping track of which words get confused with one another
insertion_table = defaultdict(dict)
deletion_table = defaultdict(dict)
substitution_table = defaultdict(dict)
ignored_count = 0
total_errors = 0
# These are the editdistance opcodes that are condsidered 'errors'
error_codes = ['replace', 'delete', 'insert']
mono_syllables = ["है","हैं","हो","हे","हूँ","हूं","हुं","हाँ","हु","हाँ","भी","जो","जी","तो","ने","ह","का","की","को","के","से","ना"]

import sys
import operator
from typing import Sequence

INSERT: str = "insert"
DELETE: str = "delete"
EQUAL: str = "equal"
REPLACE: str = "replace"


# Cost is basically: was there a match or not.
# The other numbers are cumulative costs and matches.


def lowest_cost_action(ic, dc, sc, im, dm, sm, cost) -> str:
    """Given the following values, choose the action (insertion, deletion,
    or substitution), that results in the lowest cost (ties are broken using
    the 'match' score).  This is used within the dynamic programming algorithm.
    * ic - insertion cost
    * dc - deletion cost
    * sc - substitution cost
    * im - insertion match (score)
    * dm - deletion match (score)
    * sm - substitution match (score)
    """
    best_action = None
    best_match_count = -1
    min_cost = min(ic, dc, sc)
    if min_cost == sc and cost == 0:
        best_action = EQUAL
        best_match_count = sm
    elif min_cost == sc and cost == 1:
        best_action = REPLACE
        best_match_count = sm
    elif min_cost == ic and im > best_match_count:
        best_action = INSERT
        best_match_count = im
    elif min_cost == dc and dm > best_match_count:
        best_action = DELETE
        best_match_count = dm
    else:
        raise Exception("internal error: invalid lowest cost action")
    return best_action


def highest_match_action(ic, dc, sc, im, dm, sm, cost) -> str:
    """Given the following values, choose the action (insertion, deletion, or
    substitution), that results in the highest match score (ties are broken
    using the distance values).  This is used within the dynamic programming
    algorithm.
    * ic - insertion cost
    * dc - deletion cost
    * sc - substitution cost
    * im - insertion match (score)
    * dm - deletion match (score)
    * sm - substitution match (score)
    """
    best_action = None
    lowest_cost = float("inf")
    max_match = max(im, dm, sm)
    if max_match == sm and cost == 0:
        best_action = EQUAL
        lowest_cost = sm
    elif max_match == sm and cost == 1:
        best_action = REPLACE
        lowest_cost = sm
    elif max_match == im and ic < lowest_cost:
        best_action = INSERT
        lowest_cost = ic
    elif max_match == dm and dc < lowest_cost:
        best_action = DELETE
        lowest_cost = dc
    else:
        raise Exception("internal error: invalid highest match action")
    return best_action


class SequenceMatcher(object):
    """
    Similar to the :py:mod:`difflib` :py:class:`~difflib.SequenceMatcher`, but
    uses Levenshtein/edit distance.
    """

    def __init__(
        self,
        a: Sequence = None,
        b: Sequence = None,
        test=operator.eq,
        action_function=lowest_cost_action,
    ):
        """
        Initialize the object with sequences a and b.  Optionally, one can
        specify a test function that is used to compare sequence elements. This
        defaults to the built in ``eq`` operator (i.e. :py:func:`operator.eq`).
        """
        if a is None:
            a = []
        if b is None:
            b = []
        self.seq1 = a
        self.seq2 = b
        self._reset_object()
        self.action_function = action_function
        self.test = test
        self.dist = None
        self._matches = None
        self.opcodes = None

    def set_seqs(self, a: Sequence, b: Sequence) -> None:
        """Specify two alternative sequences -- reset any cached values."""
        self.set_seq1(a)
        self.set_seq2(b)
        self._reset_object()

    def _reset_object(self) -> None:
        """Clear out the cached values for distance, matches, and opcodes."""
        self.opcodes = None
        self.dist = None
        self._matches = None

    def set_seq1(self, a: Sequence) -> None:
        """Specify a new sequence for sequence 1, resetting cached values."""
        self._reset_object()
        self.seq1 = a

    def set_seq2(self, b: Sequence) -> None:
        """Specify a new sequence for sequence 2, resetting cached values."""
        self._reset_object()
        self.seq2 = b

    def find_longest_match(self, alo, ahi, blo, bhi) -> None:
        """Not implemented!"""
        raise NotImplementedError()

    def get_matching_blocks(self):
        """Similar to :py:meth:`get_opcodes`, but returns only the opcodes that are
        equal and returns them in a somewhat different format
        (i.e. ``(i, j, n)`` )."""
        opcodes = self.get_opcodes()
        match_opcodes = filter(lambda x: x[0] == EQUAL, opcodes)
        return map(
            lambda opcode: [opcode[1], opcode[3], opcode[2] - opcode[1]], match_opcodes
        )

    def get_opcodes(self):
        """Returns a list of opcodes.  Opcodes are the same as defined by
        :py:mod:`difflib`."""
        if not self.opcodes:
            d, m, opcodes = edit_distance_backpointer(
                self.seq1,
                self.seq2,
                action_function=self.action_function,
                test=self.test,
            )
            print(d,m)
            if self.dist:
                assert d == self.dist
            if self._matches:
                assert m == self._matches
            self.dist = d
            self._matches = m
            self.opcodes = opcodes
        return self.opcodes

    def get_grouped_opcodes(self, n=None):
        """Not implemented!"""
        raise NotImplementedError()

    def ratio(self) -> float:
        """Ratio of matches to the average sequence length."""
        return 2.0 * self.matches() / (len(self.seq1) + len(self.seq2))

    def quick_ratio(self) -> float:
        """Same as :py:meth:`ratio`."""
        return self.ratio()

    def real_quick_ratio(self) -> float:
        """Same as :py:meth:`ratio`."""
        return self.ratio()

    def _compute_distance_fast(self) -> None:
        """Calls edit_distance, and asserts that if we already have values for
        matches and distance, that they match."""
        d, m = edit_distance(
            self.seq1, self.seq2, action_function=self.action_function, test=self.test
        )
        if self.dist:
            assert d == self.dist
        if self._matches:
            
            assert m == self._matches
        self.dist = d
        self._matches = m

    def distance(self):
        """Returns the edit distance of the two loaded sequences.  This should
        be a little faster than getting the same information from
        :py:meth:`get_opcodes`."""
        
        if not self.dist:
            self._compute_distance_fast()
        return self.dist

    def matches(self):
        """Returns the number of matches in the alignment of the two sequences.
        This should be a little faster than getting the same information from
        :py:meth:`get_opcodes`."""
        if not self._matches:
            self._compute_distance_fast()
        return self._matches


def edit_distance(
    seq1: Sequence, seq2: Sequence, action_function=lowest_cost_action, test=operator.eq
):
    """
    Computes the edit distance between the two given sequences.  This uses the
    relatively fast method that only constructs two columns of the 2d array
    for edits.  This function actually uses four columns because we track the
    number of matches too.
    """
    
    m = len(seq1)
    n = len(seq2)
    # Special, easy cases:
    if seq1 == seq2:
        
        return 0, n
    if m == 0:
        return n, 0
    if n == 0:
        return m, 0
    v0 = [0] * (n + 1)  # The two 'error' columns
    v1 = [0] * (n + 1)
    m0 = [0] * (n + 1)  # The two 'match' columns
    m1 = [0] * (n + 1)
    for i in range(1, n + 1):
        v0[i] = i
    for i in range(1, m + 1):
        v1[0] = i
        for j in range(1, n + 1):
            cost = 0 if test(seq1[i - 1], seq2[j - 1]) else 1
            
            # The costs
            ins_cost = v1[j - 1] + 1
            del_cost = v0[j] + 1
            sub_cost = v0[j - 1] + cost
            # Match counts
            ins_match = m1[j - 1]
            del_match = m0[j]
            sub_match = m0[j - 1] + int(not cost)

            action = action_function(
                ins_cost, del_cost, sub_cost, ins_match, del_match, sub_match, cost
            )

            if action in [EQUAL, REPLACE]:
                v1[j] = sub_cost
                m1[j] = sub_match
            elif action == INSERT:
                v1[j] = ins_cost
                m1[j] = ins_match
            elif action == DELETE:
                v1[j] = del_cost
                m1[j] = del_match
            else:
                raise Exception("Invalid dynamic programming option returned!")
                # Copy the columns over
        for k in range(n + 1):
            v0[k] = v1[k]
            m0[k] = m1[k]
    return v1[n], m1[n]


def edit_distance_backpointer(
    seq1, seq2, action_function=lowest_cost_action, test=operator.eq
):
    """
    Similar to :py:func:`~edit_distance.edit_distance` except that this
    function keeps backpointers during the search.  This allows us to return
    the opcodes (i.e. the specific edits that were used to change from one
    string to another).  This function contructs the full 2d array for the
    backpointers only.
    """
    
    m: int = len(seq1)
    n: int = len(seq2)
    # backpointer array:
    bp = [[None for x in range(n + 1)] for y in range(m + 1)]
    tp = [[None for x in range(n + 1)] for y in range(m + 1)]

            

    # Two columns of the distance and match arrays
    d0 = [0] * (n + 1)  # The two 'distance' columns
    d1 = [0] * (n + 1)
    m0 = [0] * (n + 1)  # The two 'match' columns
    m1 = [0] * (n + 1)

    # Fill in the first column
    for i in range(1, n + 1):
        d0[i] = i
        bp[0][i] = INSERT

    for i in range(1, m + 1):
        d1[0] = i
        bp[i][0] = DELETE

        for j in range(1, n + 1):

            cost = 0 if test(seq1[i - 1], seq2[j - 1]) else 1
#             if cost and (seq2[j - 1] in mono_syllables or seq1[i - 1] in mono_syllables):
                
            # The costs of each action...
            ins_cost = d1[j - 1] + 1  # insertion
            del_cost = d0[j] + 1  # deletion
            sub_cost = d0[j - 1] + cost  # substitution/match

            # The match scores of each action
            ins_match = m1[j - 1]
            del_match = m0[j]
            sub_match = m0[j - 1] + int(not cost)

            action = action_function(
                ins_cost, del_cost, sub_cost, ins_match, del_match, sub_match, cost
            )
            if action == EQUAL:
                d1[j] = sub_cost
                m1[j] = sub_match
                bp[i][j] = EQUAL
                tp[i][j] = seq1[i - 1]+"$"+seq2[j - 1]
            elif action == REPLACE:
                d1[j] = sub_cost
                m1[j] = sub_match
                bp[i][j] = REPLACE
                tp[i][j] = seq1[i - 1]+"$"+seq2[j - 1]
                
            elif action == INSERT:
                d1[j] = ins_cost
                m1[j] = ins_match
                bp[i][j] = INSERT
                tp[i][j] = ""+"$"+seq2[j - 1]
            elif action == DELETE:
                d1[j] = del_cost
                m1[j] = del_match
                bp[i][j] = DELETE
                tp[i][j] = seq1[i - 1]+"$"+" "
            else:
                raise Exception("Invalid dynamic programming action returned!")
        # copy over the columns
        for k in range(n + 1):
            d0[k] = d1[k]
            m0[k] = m1[k]
    opcodes = get_opcodes_from_bp_table(bp,tp)
#     print(opcodes)
    return d1[n], m1[n], opcodes


def get_opcodes_from_bp_table(bp,tp):
    """Given a 2d list structure, create opcodes from the best path."""
    x = len(bp) - 1
    y = len(bp[0]) - 1
    opcodes = []
    while x != 0 or y != 0:
        this_bp = bp[x][y]
        if tp[x][y]:
            tt = tp[x][y].split("$")
        
        if this_bp in [EQUAL, REPLACE]:
            opcodes.append([this_bp, max(x - 1, 0), x, max(y - 1, 0), y,tt[0],tt[1]])
            x = x - 1
            y = y - 1
        elif this_bp == INSERT:
            opcodes.append([INSERT, x, x, max(y - 1, 0), y,tt[0],tt[1]])
            y = y - 1
        elif this_bp == DELETE:
            opcodes.append([DELETE, max(x - 1, 0), x, max(y - 1, 0), max(y - 1, 0),tt[0],tt[1]])
            x = x - 1
        else:
            raise Exception("Invalid dynamic programming action in BP table!")
    opcodes.reverse()
    return opcodes

# TODO - rename this function.  Move some of it into evaluate.py?
def main(args):
    """Main method - this reads the hyp and ref files, and creates
    editdistance.SequenceMatcher objects to compute the edit distance.
    All the statistics necessary statistics are collected, and results are
    printed as specified by the command line options.
    This function doesn't not check to ensure that the reference and
    hypothesis file have the same number of lines.  It will stop after the
    shortest one runs out of lines.  This should be easy to fix...
    """
    global counter
    global ignored_count
    global total_errors

    set_global_variables(args)
    filename = ""
    counter = 0
    # Loop through each line of the reference and hyp file
    for ref_line, hyp_line in zip(args.ref, args.hyp):
        if "_norm.txt" in ref_line:
            filename = ref_line.replace("_norm.txt\n","")
            continue
        processed_p = process_line_pair(ref_line, hyp_line, filename, case_insensitive=args.case_insensitive,
                                        remove_empty_refs=args.remove_empty_refs)
        if processed_p:
            counter += 1
    if confusions:
        print_confusions()
    if wer_vs_length_p:
        print_wer_vs_length()
    # Compute WER and WRR
    if ref_token_count > 0:
        wrr = match_count / ref_token_count
        wer = error_count / ref_token_count
    else:
        wrr = 0.0
        wer = 0.0
    # Compute SER
    ser = sent_error_count / counter if counter > 0 else 0.0
    print('Sentence count: {}'.format(counter))
    print('WER: {:10.3%} ({:10d} / {:10d})'.format(wer, error_count, ref_token_count))
    print('WRR: {:10.3%} ({:10d} / {:10d})'.format(wrr, match_count, ref_token_count))
    print('SER: {:10.3%} ({:10d} / {:10d})'.format(ser, sent_error_count, counter))
    print('IGNORED: {:10d}'.format(ignored_count))
    print('ERRORS: {:10d}'.format(total_errors))




def process_line_pair(ref_line, hyp_line, filename, case_insensitive=False, remove_empty_refs=False):
    """Given a pair of strings corresponding to a reference and hypothesis,
    compute the edit distance, print if desired, and keep track of results
    in global variables.
    Return true if the pair was counted, false if the pair was not counted due
    to an empty reference string."""
    # I don't believe these all need to be global.  In any case, they shouldn't be.
    global error_count
    global match_count
    global ref_token_count
    global sent_error_count
    global ignored_count

    # Split into tokens by whitespace
    ref = ref_line.split()
    hyp = hyp_line.split()
    id_ = None

    # If the files have IDs, then split the ID off from the text
    if files_head_ids:
        id_ = ref[0]
        ref, hyp = remove_head_id(ref, hyp)
    elif files_tail_ids:
        id_ = ref[-1]
        ref, hyp = remove_tail_id(ref, hyp)

    if case_insensitive:
        ref = list(map(str.lower, ref))
        hyp = list(map(str.lower, hyp))
    if remove_empty_refs and len(ref) == 0:
        return False

    # Create an object to get the edit distance, and then retrieve the
    # relevant counts that we need.
    sm = SequenceMatcher(a=ref, b=hyp)
    errors = get_error_count(sm)
    matches = get_match_count(sm)
    ref_length = len(ref)

    # Increment the total counts we're tracking
    error_count += errors
    match_count += matches
    ref_token_count += ref_length

    if errors != 0:
        sent_error_count += 1

    # If we're keeping track of which words get mixed up with which others, call track_confusions
    if confusions:
        track_confusions(sm, ref, hyp,filename)

    # If we're printing instances, do it here (in roughly the align.c format)
    if print_instances_p or (print_errors_p and errors != 0):
        print_instances(ref, hyp, sm, id_=id_)

    # Keep track of the individual error rates, and reference lengths, so we
    # can compute average WERs by sentence length
    lengths.append(ref_length)
    error_rate = errors * 1.0 / len(ref) if len(ref) > 0 else float("inf")
    error_rates.append(error_rate)
    wer_bins[len(ref)].append(error_rate)
    return True

def set_global_variables(args):
    """Copy argparse args into global variables."""
    global print_instances_p
    global print_errors_p
    global files_head_ids
    global files_tail_ids
    global confusions
    global min_count
    global wer_vs_length_p
    # Put the command line options into global variables.
    print_instances_p = args.print_instances
    print_errors_p = args.print_errors
    files_head_ids = args.head_ids
    files_tail_ids = args.tail_ids
    confusions = args.confusions
    min_count = args.min_word_count
    wer_vs_length_p = args.print_wer_vs_length

def remove_head_id(ref, hyp):
    """Assumes that the ID is the begin token of the string which is common
    in Kaldi but not in Sphinx."""
    ref_id = ref[0]
    hyp_id = hyp[0]
    if ref_id != hyp_id:
        print('Reference and hypothesis IDs do not match! '
              'ref="{}" hyp="{}"\n'
              'File lines in hyp file should match those in the ref file.'.format(ref_id, hyp_id))
        exit(-1)
    ref = ref[1:]
    hyp = hyp[1:]
    return ref, hyp

def remove_tail_id(ref, hyp):
    """Assumes that the ID is the final token of the string which is common
    in Sphinx but not in Kaldi."""
    ref_id = ref[-1]
    hyp_id = hyp[-1]
    if ref_id != hyp_id:
        print('Reference and hypothesis IDs do not match! '
              'ref="{}" hyp="{}"\n'
              'File lines in hyp file should match those in the ref file.'.format(ref_id, hyp_id))
        exit(-1)
    ref = ref[:-1]
    hyp = hyp[:-1]
    return ref, hyp

def print_instances(ref, hyp, sm, id_=None):
    """Print a single instance of a ref/hyp pair."""
    global ignored_count
    global total_errors
    print_diff(sm, ref, hyp)
    if id_:
        print(('SENTENCE {0:d}  {1!s}'.format(counter + 1, id_)))
    else:
        print('SENTENCE {0:d}'.format(counter + 1))
    # Handle cases where the reference is empty without dying
    if len(ref) != 0:
        correct_rate = sm.matches() / len(ref)
        error_rate = sm.distance() / len(ref)
    elif sm.matches() == 0:
        correct_rate = 1.0
        error_rate = 0.0
    else:
        correct_rate = 0.0
        error_rate = sm.matches()
    print('Correct          = {0:6.1%}  {1:3d}   ({2:6d})'.format(correct_rate, sm.matches(), len(ref)))
    print('Errors           = {0:6.1%}  {1:3d}   ({2:6d})'.format(error_rate, sm.distance(), len(ref)))
    print('IGNORED:',ignored_count)
    print('ERRORs:',total_errors)
    

def track_confusions(sm, seq1, seq2,filename):
    """Keep track of the errors in a global variable, given a sequence matcher."""
    opcodes = sm.get_opcodes()
    #print(seq1)
    for tag, i1, i2, j1, j2,tt1,tt2 in opcodes:
        if tag == 'insert':
            for i in range(j1, j2):
                word = seq2[i]
                if word in mono_syllables:
                    continue
                if word in insertion_table.keys():
                    insertion_table[word]['count'] += 1
                    insertion_table[word]['files'].append(filename)
                else:
                    insertion_table[word] = {'count':1, 'files':[filename]}
                    
        elif tag == 'delete':
            for i in range(i1, i2):
                word = seq1[i]
                if word in mono_syllables:
                    continue
                if word in deletion_table.keys():
                    deletion_table[word]['count'] += 1
                    deletion_table[word]['files'].append(filename)
                else:
                    deletion_table[word] = {'count':1, 'files':[filename]}
        elif tag == 'replace':
            for w1 in seq1[i1:i2]:
                for w2 in seq2[j1:j2]:
                    key = (w1, w2)
                    if w1 in mono_syllables or w2 in mono_syllables:
                        continue
                    if key in substitution_table.keys():
                        substitution_table[key]['count'] += 1
                        substitution_table[key]['files'].append(filename)
                    else:
                        substitution_table[key] = {'count':1, 'files':[filename]}

def print_confusions():
    """Print the confused words that we found... grouped by insertions, deletions
    and substitutions."""
    
    import csv
    import pandas as pd
    insertions = []
    substitutions = []
    deletions = []
    if len(insertion_table) > 0:
        print('INSERTIONS:')
        for item in sorted(list(insertion_table.items()), key=lambda x: x[1]['count'], reverse=True):
            if item[1]['count'] >= min_count:
                print('{0:20s} {1:10d}'.format(item[0],item[1]['count']))
                insertions.append({"word":item[0], "count":item[1]['count'],"files":item[1]['files'] 
                })
    if len(deletion_table) > 0:
        print('DELETIONS:')
        for item in sorted(list(deletion_table.items()), key=lambda x: x[1]['count'], reverse=True):
            if item[1]['count'] >= min_count:
                print('{0:20s} {1:10d}'.format(item[0],item[1]['count']))
                deletions.append({"word":item[0], "count":item[1]['count'],"files":item[1]['files']
                })
    if len(substitution_table) > 0:
        print('SUBSTITUTIONS:')
        for [w1, w2], second in sorted(list(substitution_table.items()), key=lambda x: x[1]['count'], reverse=True):
            if second['count'] >= min_count:
                print('{0:20s} -> {1:20s}   {2:10d}'.format(w1, w2, second['count']))
                substitutions.append({"word":w1, "substitution":w2, "count":second['count'],"files":second['files']
                })
    pd.DataFrame(insertions).to_excel("insertions.xls",index=None)
    pd.DataFrame(deletions).to_excel("deletions.xls",index=None)
    pd.DataFrame(substitutions).to_excel("substitutions.xls",index=None)

# TODO - For some reason I was getting two different counts depending on how I count the matches,
# so do an assertion in this code to make sure we're getting matching counts.
# This might slow things down.
def get_match_count(sm):
    "Return the number of matches, given a sequence matcher object."
    matches = None
    matches1 = sm.matches()
    matching_blocks = sm.get_matching_blocks()
    matches2 = reduce(lambda x, y: x + y, [x[2] for x in matching_blocks], 0)
    assert matches1 == matches2
    matches = matches1
    return matches


def get_error_count(sm):
    """Return the number of errors (insertion, deletion, and substitutiions
    , given a sequence matcher object."""
    global ignored_count
    global total_errors
    opcodes = sm.get_opcodes()
    #print(len(opcodes))
    #[print(x) for x in opcodes if x[0] == 'replace' and x[5] in mono_syllables]
    errors = []
    for x in opcodes:
        if x[0] == 'delete' and x[5] not in mono_syllables:
            errors.append(x)
            total_errors+=1
            
        elif x[0] == 'insert' and x[6] not in mono_syllables:
            errors.append(x)
            total_errors+=1
        elif x[0] == 'replace' and not (x[5] in mono_syllables and x[6] in mono_syllables):
#             if x[5] not in mono_syllables and x[6] not in mono_syllables:
#                 errors.append(x)
#                 total_errors+=1
                
#             elif x[5] in mono_syllables and x[6] not in mono_syllables:
#                 errors.append(x)
#                 total_errors+=1
                
#             elif x[5] not in mono_syllables and x[6] in mono_syllables:
                errors.append(x)

                total_errors+=1
    
    
            
    # just print what we are ignoring  
    for x in opcodes:
        if x[0] == 'delete' and x[5] in mono_syllables:
            print(x)
            ignored_count+=1
        elif x[0] == 'insert' and x[6] in mono_syllables:
            print(x)
            ignored_count+=1
        elif x[0] == 'replace' and (x[5] in mono_syllables and x[6] in mono_syllables):
            print(x)
            ignored_count+=1
            
    
    #errors = [x for x in opcodes if x[0] in error_codes and x[5] not in mono_syllables ]
    error_lengths = [max(x[2] - x[1], x[4] - x[3]) for x in errors]
    #print(errors, error_lengths)
    return reduce(lambda x, y: x + y, error_lengths, 0)

# TODO - This is long and ugly.  Perhaps we can break it up?
# It would make more sense for this to just return the two strings...
def print_diff(sm, seq1, seq2, prefix1='REF:', prefix2='HYP:', suffix1=None, suffix2=None):
    """Given a sequence matcher and the two sequences, print a Sphinx-style
    'diff' off the two."""
    ref_tokens = []
    hyp_tokens = []
    opcodes = sm.get_opcodes()
    for tag, i1, i2, j1, j2,tt1,tt2 in opcodes:
        # If they are equal, do nothing except lowercase them
        if tag == 'equal':
            for i in range(i1, i2):
                ref_tokens.append(seq1[i].lower())
            for i in range(j1, j2):
                hyp_tokens.append(seq2[i].lower())
        # For insertions and deletions, put a filler of '***' on the other one, and
        # make the other all caps
        elif tag == 'delete':
            for i in range(i1, i2):
                if seq1[i] in mono_syllables:
                    ref_token = colored(seq1[i].upper(), 'magenta','on_yellow')
                    ref_tokens.append(ref_token)
                else:
                    ref_token = colored(seq1[i].upper(), 'red')
                    ref_tokens.append(ref_token)
            for i in range(i1, i2):
                if seq1[i] in mono_syllables:
                    hyp_token = colored('*' * len(seq1[i]), 'magenta','on_yellow')
                    hyp_tokens.append(hyp_token)
                else:
                    hyp_token = colored('*' * len(seq1[i]), 'red')
                    hyp_tokens.append(hyp_token)
        elif tag == 'insert':
            for i in range(j1, j2):
                if seq2[i] in mono_syllables:
                    ref_token = colored('*' * len(seq2[i]), 'magenta','on_cyan')
                    ref_tokens.append(ref_token)
                else:
                    ref_token = colored('*' * len(seq2[i]), 'green')
                    ref_tokens.append(ref_token)
            for i in range(j1, j2):
                if seq2[i] in mono_syllables:
                    hyp_token = colored(seq2[i].upper(), 'magenta','on_cyan')
                    hyp_tokens.append(hyp_token)
                else:
                    hyp_token = colored(seq2[i].upper(), 'green')
                    hyp_tokens.append(hyp_token)
        # More complicated logic for a substitution
        elif tag == 'replace':
            seq1_len = i2 - i1
            seq2_len = j2 - j1
            # Get a list of tokens for each
            s1 = list(map(str.upper, seq1[i1:i2]))
            s2 = list(map(str.upper, seq2[j1:j2]))
            # Pad the two lists with False values to get them to the same length
            if seq1_len > seq2_len:
                for i in range(0, seq1_len - seq2_len):
                    s2.append(False)
            if seq1_len < seq2_len:
                for i in range(0, seq2_len - seq1_len):
                    s1.append(False)
            assert len(s1) == len(s2)
            # Pair up words with their substitutions, or fillers
            for i in range(0, len(s1)):
                w1 = s1[i]
                w2 = s2[i]
                # If we have two words, make them the same length
                if w1 and w2:
                    if len(w1) > len(w2):
                        s2[i] = w2 + ' ' * (len(w1) - len(w2))
                    elif len(w1) < len(w2):
                        s1[i] = w1 + ' ' * (len(w2) - len(w1))
                # Otherwise, create an empty filler word of the right width
                if not w1:
                    s1[i] = '*' * len(w2)
                if not w2:
                    s2[i] = '*' * len(w1)
#             print(w1,w2,s1,s2)     
            if w1 in mono_syllables and w2 in mono_syllables:       
                s1 = map(lambda x: colored(x, 'magenta','on_grey'), s1)
                s2 = map(lambda x: colored(x, 'magenta','on_grey'), s2)
            else:
                s1 = map(lambda x: colored(x, 'blue'), s1)
                s2 = map(lambda x: colored(x, 'blue'), s2)
            ref_tokens += s1
            hyp_tokens += s2
    if prefix1: ref_tokens.insert(0, prefix1)
    if prefix2: hyp_tokens.insert(0, prefix2)
    if suffix1: ref_tokens.append(suffix1)
    if suffix2: hyp_tokens.append(suffix2)
    print(' '.join(ref_tokens))
    print(' '.join(hyp_tokens))

def mean(seq):
    """Return the average of the elements of a sequence."""
    return float(sum(seq)) / len(seq) if len(seq) > 0 else float('nan')

def print_wer_vs_length():
    """Print the average word error rate for each length sentence."""
    avg_wers = {length: mean(wers) for length, wers in wer_bins.items()}
    for length, avg_wer in sorted(avg_wers.items(), key=lambda x: (x[1], x[0])):
        print('{0:5d} {1:f}'.format(length, avg_wer))
    print('')

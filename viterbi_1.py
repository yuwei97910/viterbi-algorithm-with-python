"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import numpy as np

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # training step: run through all words in the training dataset
    tag_dict = {} # key: tag, value: count
    tag_init_dict = {} # key: tag, value: count if tag is the first tag in the sentence
    tag_word_dict = {} # key: tag, value: word_dict; word_dict - key: word, value: count
    tag_tag_dict = {} # key: tag, value: plus_one_tag_dict; plus_one_tag_dict - key: plus one tag, value: count

    for sentence in train:
        sentence_len = len(sentence)
        for i in range(sentence_len):
            word, tag = sentence[i]

            # update tag_dict
            tag_dict[tag] = tag_dict.get(tag, 0) + 1

            # update tag_init_dict
            if i == 0:
                tag_init_dict[tag] = tag_init_dict.get(tag, 0) + 1

            # update word_dict for the tag
            word_dict = tag_word_dict.get(tag, {})
            word_dict[word] = word_dict.get(word, 0) + 1
            tag_word_dict[tag] = word_dict

            # update plus_one_tag_dict: count tag_i then tag_i+1
            if i + 1 != sentence_len:
                tag_plus_one = sentence[i+1][1]
                plus_one_tag_dict = tag_tag_dict.get(tag, {})
                plus_one_tag_dict[tag_plus_one] = plus_one_tag_dict.get(tag_plus_one, 0) + 1
                tag_tag_dict[tag] = plus_one_tag_dict

    all_tags = list(tag_dict.keys())
    all_tags_cnt = len(all_tags)

    # Using Laplace Smoothing
    smoothing_para = 0.000001

    # generate initial prob P_s(t_1)
    init_dict = {} # key: word, value: init prob
    for tag in tag_init_dict.keys():
        init_dict[tag] = tag_init_dict[tag] / len(train)

    # generate transition prob P_T(t_k|t_(k-1))
    transition_dict = {} # key: tag; value: previous_tag_dict; previous_tag_dict - key: tag, value: prob of given previous_tag and get tag

    for tag_a in all_tags:
        temp_dict = {}
        for tag_b in all_tags:
            temp_dict[tag_b] = 0
        transition_dict[tag_a] = temp_dict

    for previous_tag in all_tags:
        for tag in all_tags:
            previous_tag_dict = tag_tag_dict.get(previous_tag, {})
            previous_tag_cnt = sum(list(previous_tag_dict.values()))
            tag_cnt = previous_tag_dict.get(tag, 0)
            if previous_tag_cnt == 0:
                transition_dict[previous_tag][tag] = np.log(smoothing_para)
            else:
                transition_dict[previous_tag][tag] = np.log((tag_cnt/previous_tag_cnt)*(1-smoothing_para) + smoothing_para)

    # generate emission prob P_E(w_i|t_i)
    emission_dict = {}
    for tag, word_dict in tag_word_dict.items():
        v = len(list(word_dict.keys()))
        n = sum(list(word_dict.values()))
        emission_tag_dict = {}
        emission_tag_dict['UNKNOWN'] = np.log(smoothing_para/(n+smoothing_para*(v+1)))
        for word, cnt in word_dict.items():
            emission_tag_dict[word] = np.log((cnt+smoothing_para)/(n+smoothing_para*(v+1)))

        emission_dict[tag] = emission_tag_dict

    # testing step: tagging
    result = []
    for sentence in test:
        if len(sentence) == 0:
            result.append([])
            continue

        v_dict, b_dict = construct_trellis(sentence, init_dict, transition_dict, emission_dict, smoothing_para)
        tagged_sentence = backtrack_trellis(sentence, v_dict, b_dict)
        result.append(list(tagged_sentence))

    return result

def construct_trellis(sentence:list, init_dict:dict, transition_dict:dict, emission_dict:dict, alpha:float):
    v_dict = {} #key: k (k>=0), value: v_dict_i; v_dict_i - key: tag, value: prob (v)
    b_dict = {} # key: k (k>=1), value: b_dict_i; b_dict_i - key: tag (all kinds of tags), value: previous tag
    for i, word in enumerate(sentence):
        if i+1 == len(sentence):
            break

        if i == 0:
            v_dict_i = {}
            for tag in emission_dict.keys():
                p_e = emission_dict[tag].get(word, emission_dict[tag]['UNKNOWN'])
                v_dict_i[tag] = init_dict.get(tag, 0) + p_e
            v_dict[0] = v_dict_i
        
        v_dict_next = {}
        b_dict_next = {}
        v_dict_i = v_dict[i]
        next_word = sentence[i+1]
        for tag_b in emission_dict.keys():
            temp_highest_tag = ''
            temp_highest_prob = -1000000
            for tag_a in v_dict_i.keys():
                p_e = emission_dict[tag_b].get(next_word, emission_dict[tag_b]['UNKNOWN'])
                p_t = transition_dict[tag_a].get(tag_b, alpha)
                temp_v = v_dict_i[tag_a] + p_t + p_e
                if float(temp_v) >= temp_highest_prob:
                    temp_highest_prob = temp_v
                    temp_highest_tag = tag_a
            v_dict_next[tag_b] = temp_highest_prob
            b_dict_next[tag_b] = temp_highest_tag

        v_dict[i+1] = v_dict_next
        b_dict[i+1] = b_dict_next
    
    return v_dict, b_dict

def backtrack_trellis(sentence:list, v_dict:dict, b_dict:dict):
    result = [''] * len(sentence)

    i = max(v_dict.keys())
    key = max(v_dict[i], key=v_dict[i].get)
    result[i] = (sentence[i], key)
    while i > 0:
        b_dict_i = b_dict[i]
        key = b_dict_i[key]

        result[i-1] = (sentence[i-1], key)
        
        i += -1

    # print(result)
    return result
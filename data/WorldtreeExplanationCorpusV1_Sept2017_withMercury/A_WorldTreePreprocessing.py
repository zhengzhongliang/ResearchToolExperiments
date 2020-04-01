import re
import itertools


RAW_TEXT_PATH = 'data/WorldtreeExplanationCorpusV1_Sept2017_withMercury/explanations_plaintext.withmercury.txt'

def check_missing_question(raw_text_list):
  ''' Although it seems there are 2200 questions in the dataset, there are only about 1600 entries. This function is used to check the missing questions in the plain text file. '''
  i=0
  j=0
  print('The following questions are missing from the plain text:')
  while i< 2201 and j< len(raw_text_list):
    question_num = int(raw_text_list[j][0][10:])
    if i!=question_num:
      print(i)
      i=question_num
    i+=1
    j+=1
  
  return 0
  
def check_generated_questions(questions, kb,  n_questions = 3):
  ''' Check if the generated questions are valid '''
  print('\n\nCHECK QUESTIONS!')
  for i in range(n_questions):
    print('='*20)
    print('id:', questions[i]['id'])
    print('stem:', questions[i]['stem'])
    print('choices:', questions[i]['choices'])
    print('facts:',questions[i]['facts'])
    for item in questions[i]['facts']:
      if type(item) is list:
        print('\t',[kb[fact_num] for fact_num in item])
      else:
        print('\t',kb[item])
    
  return 0
    
def check_generated_kb(kb, n_facts=20):
  ''' Check if the generated knowledge base is valid '''
  print('\n\nCHECK KNOWLEDGE BASE!')
  print('Num of facts in KB:', len(kb))
  print('='*20)
  for i in range(n_facts):
    print(kb[i])
    
  return 0
    
    
def match_branch_in_explanation():
  '''This function is used to check if there are any branches in the explanation. To increase the models generalization ability, we should reformat the branches in the explanation. '''

  # test_text = "(an adaptation ; an ability) has a positive impact on an (animal 's ; living thing 's) (survival ; health ; ability to reproduce) "
  
  pattern = re.compile("\([^\)]*[\;[^\)]*]*\)")  
  # explanation of this regular expression:
  # [^\)] match any character except )
  # [^\)]* the pattern [^\)] can happen no matter how many times
  # [\;[^\)]*]* the pattern [\;[^\)]*] can happen no matter how many times

  # the start, end, and pattern can be accessed by: pattern.start(), pattern.end(), pattern.group())
  matches = [(match.start(), match.end(), match.group()) for match in pattern.finditer(raw_text)]
  
  return matches
  
def format_explanation_branches(explanation, explanation_matches):
  explanations_formatted = list([])

  # Fristly we generate the segments of the fact for later combination with alternative branches.
  n_matches = len(explanation_matches)
  n_segs = len(explanation_matches)+1
  fact_segs = [' ' for i in range(n_segs+n_matches)]
  for i in range(n_segs):
    if i==0:
      fact_segs[2*i]=explanation[0:explanation_matches[0][0]]
    elif i==n_segs-1:
      fact_segs[2*i] = explanation[explanation_matches[i-1][1]:]
    else:
      fact_segs[2*i] = explanation[explanation_matches[i-1][1]:explanation_matches[i][0]]
      
  # Here we have a list of segments, where slots of alternative explanations are empty
  matches_list = [match[2][1:-1].split(';') for match in explanation_matches]
  
  explanation_seg_permutations = list(itertools.product(*matches_list))

  # Here we enumerate all possible permutations of the empty slots, and fill them to the created segments.
  for explanation_combination in explanation_seg_permutations:
    fact_segs_temp = fact_segs.copy()
    for i, token in enumerate(explanation_combination):
      fact_segs_temp[2*i+1] = token
      
    fact_seg_temp = " ".join(" ".join(fact_segs_temp).split())
    explanations_formatted.append(fact_seg_temp)
  
  # Checking the generated alternative facts, seems to be good.
  #print(explanation)
  #print(fact_segs)
  #print('*'*5)
  #print(matches_list)
  #print(explanation_seg_permutations)
  #print(explanations_formatted)
  #input('press enter to continue')
  
  return explanations_formatted

def question_list_to_dict(question_list_list):
  question_list_dict = list([])
  knowledge_base = list([])
  knowledge_base_dict = {}
  fact_id = 0

  for question_list in question_list_list:
    question_dict = {}
    question_dict['id'] = int(question_list[0][10:])
    question_stem_choice_raw = question_list[1].split('\t')
    question_dict['stem'] = question_stem_choice_raw[0][10:]
    question_dict['choices'] = [question_stem_choice_raw[i][5:] for i in range(1,5)]
    question_dict['facts'] = list([])
    
    # capture the explanations of the questions
    explanations = question_list[4:]
    for explanation in explanations:
      explanation = explanation.split('(UID')[0]
      explanation_branches = match_branch_in_explanation(explanation)
      
      # If there are branches in the explanation, process this branch and add all possible epxlanations to the reasoning path
      if len(explanation_branches)>0:
        explanations_formatted = format_explanation_branches(explanation, explanation_branches)
        # If there are branches in the explanation, there are alternative explanations in the reasoning path.
        alternative_explanations = list([])
        for explanation_formatted_single in explanations_formatted:
          if explanation_formatted_single not in knowledge_base_dict:
            knowledge_base_dict[explanation_formatted_single] = fact_id
            fact_id+=1
          alternative_explanations.append(knowledge_base_dict[explanation_formatted_single])
        question_dict['facts'].append(alternative_explanations)
        
      # If there are no branches in the explanation, just add it to the reasoning path.
      else:
        if explanation not in knowledge_base_dict:
          knowledge_base_dict[explanation] = fact_id
          fact_id+=1
        question_dict['facts'].append(knowledge_base_dict[explanation])
        
    question_list_dict.append(question_dict)
    
  # Generate the knowledge fact list from the dictionary:
  reversed_kb_dict = {v:k for k,v in knowledge_base_dict.items()}
  for i in reversed_kb_dict.keys():
    knowledge_base.append(reversed_kb_dict[i])

  return question_list_dict, knowledge_base
  
def build_dataset(debug_flag = True):
  # load from text file
  with open(RAW_TEXT_PATH,'r') as raw_file:
    raw_text = raw_file.read()
  
  raw_text = raw_text.split('\n\n\n')[:-1]   # this is to remove the \n character in the last
  
  # convert each question to a list
  question_list_list = [question.split('\n') for question in raw_text]

  # convert each question to a dictionary
  question_list_dict, knowledge_base = question_list_to_dict(question_list_list)
  
  if debug_flag:
    check_generated_questions(question_list_dict, knowledge_base)
    check_generated_kb(knowledge_base)
    
  return question_list_dict, knowledge_base
  
questions, kb = build_dataset()



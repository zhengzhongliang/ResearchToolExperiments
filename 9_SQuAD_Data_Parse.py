import json
import re
import pickle

with open("train-v1.1.json", "r") as handle:
    squad_train_json = json.load(handle)

with open("dev-v1.1.json", "r") as handle:
    squad_dev_json = json.load(handle)

def print_paragraph(paragraph):
    print("="*20)
    print("paragraph")
    for sent in paragraph.split(". "):
        print("\t"+sent)

def simple_check(sentences_dev_json):
    # sentences_dev_json has two keys: data and version
    # data has 48 elements
    # each data element has two keys: title and paragraphs
    # data 1 has 54 paragraphs
    # each paragraph has two keys: context and qas, context is a string and qas is a list
    # qas has fields "answers",

    # print_paragraph(sentences_dev_json["data"][0]["paragraphs"][0]["context"])
    # print(sentences_dev_json["data"][0]["paragraphs"][0]["qas"])
    question_0_context = sentences_dev_json["data"][0]["paragraphs"][0]["context"]
    question_0_ques = sentences_dev_json["data"][0]["paragraphs"][0]["qas"][0]["question"]
    question_0_answer_0_start = int(sentences_dev_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"])
    question_0_answer_0_text = sentences_dev_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    print("question 0 question:", question_0_ques)
    print("answer accessed:", question_0_context[question_0_answer_0_start:question_0_answer_0_start+50])
    print("true answer:", question_0_answer_0_text)

    return 0



# See this repo about how to convert the squad and nq data to retrieval dataset.
# https://github.com/google/retrieval-qa-eval/blob/master/sb_sed.py

def infer_sentence_breaks(uni_text):
  """Generates (start, end) pairs demarking sentences in the text.
  Args:
    uni_text: A (multi-sentence) passage of text, in Unicode.
  Yields:
    (start, end) tuples that demarcate sentences in the input text. Normal
    Python slicing applies: the start index points at the first character of
    the sentence, and the end index is one past the last character of the
    sentence.
  """
  # Treat the text as a single line that starts out with no internal newline
  # characters and, after regexp-governed substitutions, contains internal
  # newlines representing cuts between sentences.
  uni_text = re.sub(r'\n', r' ', uni_text)  # Remove pre-existing newlines.
  text_with_breaks = _sed_do_sentence_breaks(uni_text)
  starts = [m.end() for m in re.finditer(r'^\s*', text_with_breaks, re.M)]
  sentences = [s.strip() for s in text_with_breaks.split('\n')]
  assert len(starts) == len(sentences)
  for i in range(len(sentences)):
    start = starts[i]
    end = start + len(sentences[i])
    yield start, end


def _sed_do_sentence_breaks(uni_text):
  """Uses regexp substitution rules to insert newlines as sentence breaks.
  Args:
    uni_text: A (multi-sentence) passage of text, in Unicode.
  Returns:
    A Unicode string with internal newlines representing the inferred sentence
    breaks.
  """

  # The main split, looks for sequence of:
  #   - sentence-ending punctuation: [.?!]
  #   - optional quotes, parens, spaces: [)'" \u201D]*
  #   - whitespace: \s
  #   - optional whitespace: \s*
  #   - optional opening quotes, bracket, paren: [['"(\u201C]?
  #   - upper case letter or digit
  txt = re.sub(r'''([.?!][)'" %s]*)\s(\s*[['"(%s]?[A-Z0-9])''' % ('\u201D', '\u201C'),
               r'\1\n\2',
               uni_text)

  # Wiki-specific split, for sentence-final editorial scraps (which can stack):
  #  - ".[citation needed]", ".[note 1] ", ".[c] ", ".[n 8] "
  txt = re.sub(r'''([.?!]['"]?)((\[[a-zA-Z0-9 ?]+\])+)\s(\s*['"(]?[A-Z0-9])''',
               r'\1\2\n\4', txt)

  # Wiki-specific split, for ellipses in multi-sentence quotes:
  # "need such things [...] But"
  txt = re.sub(r'(\[\.\.\.\]\s*)\s(\[?[A-Z])', r'\1\n\2', txt)

  # Rejoin for:
  #   - social, military, religious, and professional titles
  #   - common literary abbreviations
  #   - month name abbreviations
  #   - geographical abbreviations
  #
  txt = re.sub(r'\b(Mrs?|Ms|Dr|Prof|Fr|Rev|Msgr|Sta?)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b(Lt|Gen|Col|Maj|Adm|Capt|Sgt|Rep|Gov|Sen|Pres)\.\n',
               r'\1. ',
               txt)
  txt = re.sub(r'\b(e\.g|i\.?e|vs?|pp?|cf|a\.k\.a|approx|app|es[pt]|tr)\.\n',
               r'\1. ',
               txt)
  txt = re.sub(r'\b(Jan|Aug|Oct|Nov|Dec)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b(Mt|Ft)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b([ap]\.m)\.\n(Eastern|EST)\b', r'\1. \2', txt)

  # Rejoin for personal names with 3,2, or 1 initials preceding the last name.
  txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3 \4',
               txt)
  txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3',
               txt)
  txt = re.sub(r'\b([A-Z]\.[A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)
  txt = re.sub(r'\b([A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)

  # Resplit for common sentence starts:
  #   - The, This, That, ...
  #   - Meanwhile, However,
  #   - In, On, By, During, After, ...
  txt = re.sub(r'([.!?][\'")]*) (The|This|That|These|It) ', r'\1\n\2 ', txt)
  txt = re.sub(r'(\.) (Meanwhile|However)', r'\1\n\2', txt)
  txt = re.sub(r'(\.) (In|On|By|During|After|Under|Although|Yet|As |Several'
               r'|According to) ',
               r'\1\n\2 ',
               txt)

  # Rejoin for:
  #   - numbered parts of documents.
  #   - born, died, ruled, circa, flourished ...
  #   - et al (2005), ...
  #   - H.R. 2000
  txt = re.sub(r'\b([Aa]rt|[Nn]o|Opp?|ch|Sec|cl|Rec|Ecl|Cor|Lk|Jn|Vol)\.\n'
               r'([0-9IVX]+)\b',
               r'\1. \2',
               txt)
  txt = re.sub(r'\b([bdrc]|ca|fl)\.\n([A-Z0-9])', r'\1. \2', txt)
  txt = re.sub(r'\b(et al)\.\n(\(?[0-9]{4}\b)', r'\1. \2', txt)
  txt = re.sub(r'\b(H\.R\.)\n([0-9])', r'\1 \2', txt)

  # SQuAD-specific joins.
  txt = re.sub(r'(I Am\.\.\.)\n(Sasha Fierce|World Tour)', r'\1 \2', txt)
  txt = re.sub(r'(Warner Bros\.)\n(Records|Entertainment)', r'\1 \2', txt)
  txt = re.sub(r'(U\.S\.)\n(\(?\d\d+)', r'\1 \2', txt)
  txt = re.sub(r'\b(Rs\.)\n(\d)', r'\1 \2', txt)

  # SQuAD-specific splits.
  txt = re.sub(r'\b(Jay Z\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(Washington, D\.C\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(for 4\.\)) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(Wii U\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\. (iPod|iTunes)', r'.\n\1', txt)
  txt = re.sub(r' (\[\.\.\.\]\n)', r'\n\1', txt)
  txt = re.sub(r'(\.Sc\.)\n', r'\1 ', txt)
  txt = re.sub(r' (%s [A-Z])' % '\u2022', r'\n\1', txt)
  return txt

#simple_check(sentences_dev_json)

# The following script comes from eval squad

def generate_examples(data):
    '''
    This function is used for get the question, answer sentence and the context for the whole dataset.
    :param data: json converted to dictionary
    :return:
    '''
    counts = [0, 0, 0]
    for passage in data["data"]:
        counts[0] += 1
        for paragraph in passage["paragraphs"]:
            counts[1] += 1

            paragraph_text = paragraph["context"]
            sentence_breaks = list(infer_sentence_breaks(paragraph_text))

            para = [str(paragraph_text[start:end]) for (start, end) in sentence_breaks]

            for qas in paragraph["qas"]:
                # The answer sentences that have been output for the current question.
                answer_sentences = set()  # type: Set[str]

                counts[2] += 1
                for answer in qas["answers"]:
                    answer_start = answer["answer_start"]
                    # Map the answer fragment back to its enclosing sentence.
                    sentence = None
                    for start, end in sentence_breaks:
                        if start <= answer_start < end:
                            sentence = paragraph_text[start:end]
                            break

                    # Avoid generating duplicate answer sentences.
                    if sentence not in answer_sentences:
                        answer_sentences.add(str(sentence))
                        yield (qas["question"], qas["id"], str(sentence), paragraph_text, para, answer["text"])


def convert_squad_to_retrieval(squad_json_train, squad_json_dev):
    '''
    This function is use for converting the original squad data to retrieval data.
    Note that the sentences and contexts are shared by the train set and the dev set.
    :param squad_json_train: original json file of squad train
    :param squad_json_dev: original json file of squad dev
    :return: squad_retrieval_train, a list of dicts. [{id:str, question:str, answer_sent:int, context:int, label:int, gold_answer_text:str}, {...}, ...]
    :return: squad_retrieval_dev, a list of dicts. [{id:str, question:str, answer_sent:int, context:int, label:int}, {...}, ...]
    :return: sentences: list of sentences, obtained by breaking down the context using the infer_sentence_breaks function
    :return: contexts: list of contexts, obtained by reading the context of each paragraph.
    :return: response: list of [sent_num, context_num]. This reponse list is the final list to generate the embeddings.
    '''

    seen_sentences_dict = {}
    seen_documents_dict = {}
    seen_responses_dict ={}
    sent_doc_resp_count = {"sent":0, "doc":0, "resp":0}


    squad_retrieval_train = []
    for question, q_id, answer_sent, document, para, gold_answer_text in generate_examples(squad_json_train):
        if document in seen_documents_dict:
            document_id = seen_documents_dict[document]
        else:
            document_id = sent_doc_resp_count["doc"]
            seen_documents_dict[document] = document_id
            sent_doc_resp_count["doc"]+=1

        for sentence in para:
            if sentence in seen_sentences_dict:
                sentence_id = seen_sentences_dict[sentence]
            else:
                sentence_id = sent_doc_resp_count["sent"]
                seen_sentences_dict[sentence] = sentence_id
                sent_doc_resp_count["sent"] += 1

            response_key = str(sentence_id)+","+str(document_id)
            if response_key not in seen_responses_dict:
                seen_responses_dict[response_key] = sent_doc_resp_count["resp"]
                sent_doc_resp_count["resp"]+=1

        assert(answer_sent in seen_sentences_dict)
        answer_sent_id = seen_sentences_dict[answer_sent]
        answer_doc_id = document_id
        response_id = seen_responses_dict[str(answer_sent_id)+","+str(answer_doc_id)]

        question_dict_to_append= {"id":q_id,"question":question,"answer":answer_sent_id,"document":answer_doc_id,"response":response_id, "gold_answer_text":gold_answer_text}
        squad_retrieval_train.append(question_dict_to_append)

    squad_retrieval_dev = []
    for question, q_id, answer_sent, document, para, gold_answer_text in generate_examples(squad_json_dev):
        if document in seen_documents_dict:
            document_id = seen_documents_dict[document]
        else:
            document_id = sent_doc_resp_count["doc"]
            seen_documents_dict[document] = document_id
            sent_doc_resp_count["doc"] += 1

        for sentence in para:
            if sentence in seen_sentences_dict:
                sentence_id = seen_sentences_dict[sentence]
            else:
                sentence_id = sent_doc_resp_count["sent"]
                seen_sentences_dict[sentence] = sentence_id
                sent_doc_resp_count["sent"] += 1

            response_key = str(sentence_id) + "," + str(document_id)
            if response_key not in seen_responses_dict:
                seen_responses_dict[response_key] = sent_doc_resp_count["resp"]
                sent_doc_resp_count["resp"] += 1

        assert (answer_sent in seen_sentences_dict)
        answer_sent_id = seen_sentences_dict[answer_sent]
        answer_doc_id = document_id
        response_id = seen_responses_dict[str(answer_sent_id) + "," + str(answer_doc_id)]

        question_dict_to_append = {"id": q_id, "question": question, "answer": answer_sent_id,
                                   "document": answer_doc_id, "response": response_id, "gold_answer_text":gold_answer_text}

        squad_retrieval_dev.append(question_dict_to_append)


    # convert sent dict, doc dict and reponse dict to lists:
    sent_list = [k for k,v in  sorted(seen_sentences_dict.items(), key=lambda x: x[1])]
    doc_list = [k for k,v in  sorted(seen_documents_dict.items(), key=lambda x: x[1])]
    response_list = [(k.split(",")[0], k.split(",")[1]) for k,v in  sorted(seen_responses_dict.items(), key=lambda x: x[1])]

    print("data generation finished!")


    with open("squad_retrieval_data.pickle", "wb" ) as handle:
        pickle.dump({"train_list": squad_retrieval_train,
                     "dev_list": squad_retrieval_dev,
                     "sent_list":sent_list,
                     "doc_list":doc_list,
                     "resp_list":response_list}, handle)

    return squad_retrieval_train, squad_retrieval_dev, sent_list, doc_list, response_list

#convert_squad_to_retrieval(squad_train_json, squad_dev_json)

def check_squad_retrieval_pickle():
    with open("squad_retrieval_data.pickle", "rb") as handle:
        squad_retrieval_data = pickle.load(handle)

    non_standard_example = 0
    for i, question_dict in enumerate(squad_retrieval_data["train_list"]):
        try:
            assert (question_dict["gold_answer_text"] in squad_retrieval_data["sent_list"][question_dict["answer"]])
            assert (squad_retrieval_data["sent_list"][question_dict["answer"]] in squad_retrieval_data["doc_list"][question_dict["document"]])
        except:
            # print("=" * 20)
            # print("\tquestion:", question_dict["question"])
            # print("\tanswer sent recon:", squad_retrieval_data["sent_list"][question_dict["answer"]])
            # print("\tanswer doc recon:", squad_retrieval_data["doc_list"][question_dict["document"]])
            # print("\tgold answer:", question_dict["gold_answer_text"])
            #
            # input("Strange example!!")
            non_standard_example+=1

        if i>500 and i<510:
            print("="*20)
            print("\tquestion:", question_dict["question"])
            print("\tanswer sent recon:", squad_retrieval_data["sent_list"][question_dict["answer"]])
            print("\tanswer doc recon:", squad_retrieval_data["doc_list"][question_dict["document"]])
            print("\tgold answer:", question_dict["gold_answer_text"])

    for i, question_dict in enumerate(squad_retrieval_data["dev_list"]):
        try:
            assert (question_dict["gold_answer_text"] in squad_retrieval_data["sent_list"][question_dict["answer"]])
            assert (squad_retrieval_data["sent_list"][question_dict["answer"]] in squad_retrieval_data["doc_list"][
                question_dict["document"]])
        except:
            # print("=" * 20)
            # print("\tquestion:", question_dict["question"])
            # print("\tanswer sent recon:", squad_retrieval_data["sent_list"][question_dict["answer"]])
            # print("\tanswer doc recon:", squad_retrieval_data["doc_list"][question_dict["document"]])
            # print("\tgold answer:", question_dict["gold_answer_text"])
            #
            # input("Strange example!!")
            non_standard_example+=1

        if i>500 and i<510:
            print("="*20)
            print("\tquestion:", question_dict["question"])
            print("\tanswer sent recon:", squad_retrieval_data["sent_list"][question_dict["answer"]])
            print("\tanswer doc recon:", squad_retrieval_data["doc_list"][question_dict["document"]])
            print("\tgold answer:", question_dict["gold_answer_text"])

    print("total number of broken sentences:", non_standard_example)
    print("total number of facts:", len(squad_retrieval_data["sent_list"]))
    print("total number of paragraphs:", len(squad_retrieval_data["doc_list"]))
    print("total number of examples:", len(squad_retrieval_data["dev_list"])+len(squad_retrieval_data["train_list"]))


    return 0

check_squad_retrieval_pickle()

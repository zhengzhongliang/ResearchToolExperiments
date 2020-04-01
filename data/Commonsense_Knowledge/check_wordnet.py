import json
import time
    
kb_facts_list = list([])
with open('cn5_wordnet.json', 'r', encoding='utf-8') as dataset:
    start_time = time.time()
    for i, line in enumerate(dataset):
        item = json.loads(line.strip())
        raw_text = item["surfaceText"]
        processed_text = raw_text.replace("[[","").replace("]]","")
        
        kb_facts_list.append(processed_text)
        if i%2000==0:
            print("processing sample ",i, '\t time:', time.time()-start_time)
            start_time = time.time()
            
with open('cn5_wordnet.txt', 'w') as f:
    for item in kb_facts_list:
        f.write("%s\n" % item)
            

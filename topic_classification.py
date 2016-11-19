import re
import os
import sys
import time
import glob
import pandas as pd
import numpy as np
import multiprocessing.pool as mp
import threading as td

CLEANUP_REGEX = re.compile("(^Path: )|(^From: )|(^Newsgroups: )|(^Date: )|(^Organization: )|(^Lines: )|(^Message-ID: )|(^References: )|(^NNTP-Posting-Host: )")
WORD_REGEX = re.compile("[a-zA-Z]{5,15}[,|.]*$")
CWD = os.getcwd()
WAIT_TIME = 30
#INPUT_DIR = "cat1"
#LABEL = "1"
PROCESSED_INPUT_DIR = "processed-test"
OUTPUT_DIR = "pre_svm-test-5-n2"
N_GRAM = 1

def processInputFiles():
    #lfiles = glob.glob(os.path.join(INPUT_DIR, "*"))
    lfiles = os.listdir(INPUT_DIR)
    #print lfiles
    
    for lfile in lfiles:
        with open(os.path.join(INPUT_DIR, lfile)) as f:
            print "Processing input file for category %s, file %s.. " % (INPUT_DIR, lfile)
            words = [word.lower() for line in f for word in line.split() if not re.match(CLEANUP_REGEX, line)]
            words = [word.replace(">","") for word in words]
            words = [word.replace("<","") for word in words]
            words = [word.replace("\"","") for word in words]
            words = [word.replace("'","") for word in words]
            words = [word.replace("(","") for word in words]
            words = [word.replace(")","") for word in words]
            words = [word.replace("!","") for word in words]
            words = [word.replace("?","") for word in words]
            words = [word.replace(",","") for word in words]
            words = [word.replace(".","") for word in words]
            words = filter(WORD_REGEX.match, words)
            f1= open(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR,lfile), "w")
            f1.writelines(["%s\n" % index for index in words])
            f1.close()
        f.close()

def getFeatures(lfile, return_count=False):
    n_words = []
    with open(lfile) as f:
        temp = f.read().splitlines()
        #put here further preprocessing e.g. eliminate words < 3, common words, etc
    f.close()
    #print temp
    i = 0
    while i < len(temp)-(N_GRAM-1):
        n_words.append(" ".join(temp[i:i+N_GRAM]))
        i+=1  
    #print n_words
    #uncomment this if you want 1/0
    #n_words = list(set(n_words))
    #if not n_words:
    #    return pd.DataFrame()
    if return_count:
        temp_df = pd.DataFrame([[word] for word in n_words ])
        try:
            temp_df_1 = temp_df[0].value_counts()
        except Exception:
            return pd.DataFrame()
        return temp_df_1
    return n_words
def getNgramFeatures(n_words):
    lfiles = os.listdir(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR))
    for lfile in lfiles:
        n_words += getFeatures(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR,lfile))
    #unique-fy list 
    n_words = list(set(n_words))
    #print n_words
    return n_words
  
def constructTable(feature_list, l_input_dir, l_label):
    pri_index_table = pd.DataFrame(index=feature_list)
    
    
    lfiles = os.listdir(os.path.join(PROCESSED_INPUT_DIR, l_input_dir))
    for lfile in lfiles:
        print "Populating table for category %s, file %s " % (l_input_dir, lfile)
        #file_feature_list before!
        sec_index_table = getFeatures(os.path.join(PROCESSED_INPUT_DIR, l_input_dir,lfile), True)
        #uncomment this if 0/1
        #sec_index_table = pd.DataFrame([1]*len(file_feature_list), index=file_feature_list)
        pri_index_table = pd.concat([pri_index_table, sec_index_table], axis=1)
        
    pri_index_table.fillna(0, inplace=True)
    pri_index_table.columns = [range(0, pri_index_table.shape[1])]
    #print pri_index_table
    pri_index_table.to_csv(os.path.join(PROCESSED_INPUT_DIR, "indexes-" + l_label), header=None)
    constructSVMInputFiles(pri_index_table, l_label)
    #return pri_index_table
    
def constructSVMInputFiles(pri_index_table, l_label):
    f = open(os.path.join(OUTPUT_DIR, l_label+".libsvm"), "w")
    print "Creating SVM input file for label %s.. " % (l_label)
    for i in range(0, pri_index_table.shape[1]):
        temp = [l_label]
        print pri_index_table.shape[0]
        for x, y in zip(range(1, pri_index_table.shape[0]+1), pri_index_table.ix[:, i].tolist()):
            temp.append(str(x) + ":" + str(y))
        

        f.writelines(" ".join(temp) + "\n")
    f.close()
def main():
# processing for test files is different, extract index from a separate file!
#INPUT_DIR = "cat1"
#LABEL = "1"
    cat_list = [
        "alt.atheism", 
        "comp.graphics", 
        "comp.os.ms-windows.misc", 
        "comp.sys.ibm.pc.hardware", 
        "comp.sys.mac.hardware", 
        "comp.windows.x", 
        "misc.forsale", 
        "rec.autos", 
        "rec.motorcycles", 
        "rec.sport.baseball", 
        "rec.sport.hockey", 
        "sci.crypt", 
        "sci.electronics", 
        "sci.med", 
        "sci.space", 
        "soc.religion.christian", 
        "talk.politics.guns", 
        "talk.politics.mideast",
        "talk.politics.misc", 
        "talk.religion.misc"
     ]
    cat_list = ["alt.atheism"]
    #cat_list = ["cat1", "cat2"]
    label_list = 1
    global INPUT_DIR
    global LABEL
    n_words = []
    # Uncomment this if you want to process new input files
    label_list = 1
    for cat in cat_list:
        print "Processing input file for category %s.. " % (cat)
        INPUT_DIR = cat
        LABEL = str(label_list)
        os.mkdir(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR)) 
        processInputFiles()
        label_list+=1
        
    return
    label_list = 1
    for cat in cat_list:
        print "Getting primary features for category %s.. " % (cat)
        INPUT_DIR = cat
        LABEL = str(label_list)     
        n_words = getNgramFeatures(n_words)
        label_list+=1
    feature_list = n_words
    #test = pd.DataFrame(index=feature_list)
    f1= open(os.path.join(PROCESSED_INPUT_DIR, "indexes"), "w")
    f1.writelines(["%s\n" % index for index in feature_list])
    f1.close()
    #label_list = 1
    #            train_p = mp.Process(target=training, args=(lfile,lfile,procfile,False,))
    #        train_jobs.append(train_p)
    #        train_p.start()
    #        train_p.start()
    # uncomment this for populating tables
    ##delay = 0
    ##label_list = 1
    ##for cat in cat_list:
    ##    if delay >= 3:
    ##            time.sleep(WAIT_TIME)
    ##            delay = 0
    ##    print "Populating table for category %s.. " % (cat)
    ##   INPUT_DIR = cat
    ##    LABEL = str(label_list)  
    ##    constructTable_p = mp.Process(target=constructTable, args=(feature_list,INPUT_DIR, LABEL))
    ##    constructTable_p.start()
    ##    label_list+=1
    ##    delay +=1   
    #label_list = 1
    #for cat in cat_list:
        #os.mkdir(os.path.join(OUTPUT_DIR, INPUT_DIR))
    #    print "Creating SVM input file for category %s.. " % (cat)
    #    INPUT_DIR = cat
    #    LABEL = str(label_list)  
    #    constructSVMInputFiles(pri_index_table)
    #    label_list+=1
    

if __name__ == '__main__':
    main()
    
    


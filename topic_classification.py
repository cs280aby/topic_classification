import re
import os
import sys
import time
import glob
import pandas as pd
import numpy as np
import multiprocessing.pool as mp
from collections import Counter

CLEANUP_REGEX = re.compile("(^Path: )|(^From: )|(^Newsgroups: )|(^Date: )|(^Organization: )|(^Lines: )|(^Message-ID: )|(^References: )|(^NNTP-Posting-Host: )")
WORD_REGEX = re.compile("[a-zA-Z]{5,15}[,|.]*$")
CWD = os.getcwd()
WAIT_TIME = 30
TESTING = True
# set BINARY to True if you wish output to be 0/1 instead of word frequency count
BINARY = False
# parameters for input file processing
# change PROCESSED_INPUT_DIR to the directoy where you want to place the processed input files
PROCESSED_INPUT_DIR = "processed-test-n2"
# setting PRE_PROCESS_INPUT = True will process the intput files AND proceed to produce SVM input files using the 
#    in INDEX_FILE and N_GRAM configuration
PRE_PROCESS_INPUT = False
N_GRAM = 2

# parameters for creating svm input files based on given features
# use this for pool of N-gram features
GET_INDEX = False
MIN_WORD_FREQUENCY = 50
INDEX_FILE = "indexes_n_all.txt"
# this contains the directories of the processed files based on given N-gram
PROCESSED_INPUT_DIR_N = [
    "processed-test-n1", 
    "processed-test-n2",
    "processed-test-n3" 
]
# change OUTPUT_DIR to the directoy where you want to place the svm input files
OUTPUT_DIR = "pre_svm-test-4-1"
# Set ALL_N to True if all input files for all N are already procesed
ALL_N = False

def processInputFiles(sub_dir_list):
    #lfiles = glob.glob(os.path.join(INPUT_DIR, "*"))
    for sub_dir in sub_dir_list:
        lfiles = os.listdir(os.path.join(INPUT_DIR, sub_dir))
        #print lfiles
        if not os.path.exists(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR, sub_dir)):
            os.mkdir(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR, sub_dir))
        for lfile in lfiles:
            with open(os.path.join(INPUT_DIR, sub_dir, lfile)) as f:
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
                i = 0
                n_words = []
                while i < len(words)-(N_GRAM-1):
                    n_words.append(" ".join(words[i:i+N_GRAM]))
                    i+=1  
                f1 = open(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR,sub_dir, lfile), "w")
                f1.writelines(["%s\n" % index for index in n_words])
                f1.close()
            f.close()

def getFeaturesN(lfile, feature_list, category, sub_dir):
    n_words = []
    for processed_dir in PROCESSED_INPUT_DIR_N:
        print lfile
        lfile=os.path.join(processed_dir, category, sub_dir, os.path.basename(lfile))
        with open(lfile) as f:
            temp = f.read().splitlines()
        n_words += temp
        f.close()

   
   
    # indexes must be filtered to the chosen features during training
    n_words = [word for word in n_words if word in feature_list]
    print n_words
    temp_df = pd.DataFrame([[word] for word in n_words ])
    try:
        temp_df_1 = temp_df[0].value_counts()
    except Exception:
        return pd.DataFrame()
    if BINARY:
        temp_df_1[temp_df_1!=0] = 1
        
    return temp_df_1

def getFeatures(lfile, return_count=False, feature_list=[]):
    n_words = []
    with open(lfile) as f:
        n_words = f.read().splitlines()
    f.close()
   
    if return_count:
        # indexes must be filtered to the chosen features during training
        n_words = [word for word in n_words if word in feature_list]
        temp_df = pd.DataFrame([[word] for word in n_words ])
        try:
            temp_df_1 = temp_df[0].value_counts()
        except Exception:
            return pd.DataFrame()
        if BINARY:
            temp_df_1[temp_df_1!=0] = 1
        
        return temp_df_1
    return n_words
    
def getNgramFeatures(n_words, sub_dir_list):
    for sub_dir in sub_dir_list:
        lfiles = os.listdir(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR, sub_dir))
        for lfile in lfiles:
            n_words += getFeatures(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR, sub_dir,lfile))
    #unique-fy list 
    n_words = list(set(n_words))
    return n_words
    
def constructTableSubdir(feature_list, l_input_dir, l_label, sub_dir, lfiles, pri_index_table):
    for lfile in lfiles:
        print "Populating table for category %s, sub-directory %s, file %s " % (l_input_dir, sub_dir, lfile)
        sec_index_table = getFeatures(os.path.join(PROCESSED_INPUT_DIR, l_input_dir,sub_dir,lfile), True, feature_list)
        pri_index_table = pd.concat([pri_index_table, sec_index_table], axis=1)
        
    pri_index_table.fillna(0, inplace=True)
    pri_index_table.columns = [range(0, pri_index_table.shape[1])]
    print "label: %s sub_dir: %s" % (l_label, sub_dir)
    print pri_index_table    
    constructSVMInputFiles(pri_index_table, l_label, sub_dir)

    
def constructTable(feature_list, l_input_dir, l_label, sub_dir_list):
    pri_index_table = pd.DataFrame(index=feature_list)
    
    for sub_dir in sub_dir_list:
        lfiles = os.listdir(os.path.join(PROCESSED_INPUT_DIR, l_input_dir, sub_dir))
        constructTableSubdir_p = mp.Process(target=constructTableSubdir, args=(feature_list, l_input_dir, l_label, sub_dir, lfiles, pri_index_table))
        constructTableSubdir_p.start()
        
def constructTableAllN(feature_list, l_input_dir, l_label, sub_dir_list ):
    pri_index_table = pd.DataFrame(index=feature_list)
    
    for sub_dir in sub_dir_list:
        for lfile in os.listdir(os.path.join(PROCESSED_INPUT_DIR,l_input_dir, sub_dir)):
            print "Populating table for category %s, sub-directory %s, file %s using N %d" % (l_input_dir, sub_dir, lfile, N_GRAM)

            sec_index_table = getFeaturesN(lfile, feature_list, l_input_dir, sub_dir)
            pri_index_table = pd.concat([pri_index_table, sec_index_table], axis=1)
        
    pri_index_table.fillna(0, inplace=True)
    pri_index_table.columns = [range(0, pri_index_table.shape[1])]
    print "label: %s sub_dir: %s" % (l_label, sub_dir)
    print pri_index_table
    constructSVMInputFiles(pri_index_table, l_label, sub_dir)

    
    
    
    
    
def constructSVMInputFiles(pri_index_table, l_label, sub_dir):
    f = open(os.path.join(OUTPUT_DIR, l_label+ "-" + sub_dir +".libsvm"), "w")
    print "Creating SVM input file for label %s sub-directory %s.. " % (l_label, sub_dir)
    for i in range(0, pri_index_table.shape[1]):
        temp = [l_label]
        print pri_index_table.shape[0]
        for x, y in zip(range(1, pri_index_table.shape[0]+1), pri_index_table.ix[:, i].tolist()):
            temp.append(str(x) + ":" + str(y))
        

        f.writelines(" ".join(temp) + "\n")
    f.close()
    
def preProcessInputFiles(cat_list, sub_dir_list):
    global INPUT_DIR
    global LABEL
    label_list = 1
    for cat in cat_list:
        print "Processing input file for category %s.. " % (cat)
        INPUT_DIR = cat
        LABEL = str(label_list)
        if not os.path.exists(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR)):
            os.mkdir(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR)) 
        processInputFiles(sub_dir_list)
        label_list+=1
        
def preGetFeatures(cat_list, sub_dir_list):
    global INPUT_DIR
    global LABEL
    n_words = []
    label_list = 1
    for cat in cat_list:
        print "Getting primary features for category %s.. " % (cat)
        INPUT_DIR = cat
        LABEL = str(label_list)     
        n_words = getNgramFeatures(n_words, sub_dir_list)
        label_list+=1
    feature_list = n_words

    f= open(os.path.join(PROCESSED_INPUT_DIR, "indexes_n_" + str(N_GRAM)), "w")
    f.writelines(["%s\n" % index for index in feature_list])
    f.close()    
    
def preConstructTable(cat_list, feature_list, sub_dir_list):
    global INPUT_DIR
    global LABEL
    delay = 0
    label_list = 1
    for cat in cat_list:
        if delay >= 3:
                time.sleep(WAIT_TIME)
                delay = 0
        print "Populating table for category %s.. " % (cat)
        INPUT_DIR = cat
        LABEL = str(label_list)
        if ALL_N:
            constructTable_pn = mp.Process(target=constructTableAllN, args=(feature_list,INPUT_DIR, LABEL, sub_dir_list))
            constructTable_pn.start()
        else:
            constructTable_p = mp.Process(target=constructTable, args=(feature_list,INPUT_DIR, LABEL, sub_dir_list))
            constructTable_p.start()
        label_list+=1
        delay +=1 
        

        
def main():

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
    #cat_list = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc"]
    #cat_list = ["cat1", "cat2"]
    # divide each input file folder to 4 sub-folders, 250 files each for multiprocessing
    # use folders 1-3 for training, folder 4 for testing
    # edit sub_dir_list below to indicate list of folders to be processed
    sub_dir_list = ["1", "2", "3"]
    if not os.path.exists(PROCESSED_INPUT_DIR):
        os.mkdir(PROCESSED_INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

        
    if TESTING:
        sub_dir_list = ["4"]
        preProcessInputFiles(cat_list, sub_dir_list)

    if PRE_PROCESS_INPUT:
        preProcessInputFiles(cat_list, sub_dir_list)
        preGetFeatures(cat_list, sub_dir_list)
        
    if GET_INDEX:
        print "Getting indexes above minimum frequency.."
        words = []
        for category in cat_list:
            for sub_dir in sub_dir_list:
                lfiles = glob.glob(os.path.join(PROCESSED_INPUT_DIR, category,sub_dir, "*"))
                for lfile in lfiles:
                    with open(lfile) as f:
                        temp = f.read().splitlines()
                    words+=temp
                    f.close()
                
        feature_counts = Counter(words)
        feature_list =  [word for word, count in feature_counts.items() if count > MIN_WORD_FREQUENCY]
        print "New feature vector length: %d" % (len(feature_list))
        f = open(os.path.join(INDEX_FILE), "w")
        f.writelines(["%s\n" % feature for feature in feature_list])
        f.close()
        return
    
    with open(INDEX_FILE) as f:
        feature_list = f.read().splitlines()
    time.sleep(10)
    print "Feature vector length: %d" % (len(feature_list))
    
    preConstructTable(cat_list, feature_list, sub_dir_list)
     

if __name__ == '__main__':
    main()
    
    

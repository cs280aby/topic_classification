
import time
import glob
import pandas as pd
import numpy as np
import multiprocessing.pool as mp
from collections import Counter
import argparse

CLEANUP_REGEX = re.compile("(^Path: )|(^From: )|(^Newsgroups: )|(^Date: )|(^Organization: )|(^Lines: )|(^Message-ID: )|(^References: )|(^NNTP-Posting-Host: )")
WORD_REGEX = re.compile("[a-zA-Z]{5,15}[,|.]*$")
CWD = os.getcwd()
WAIT_TIME = 30

# this contains the directories of the processed files based on given N-gram
PROCESSED_INPUT_DIR_N = [
    "processed-test-n1", 
    "processed-test-n2",
    "processed-test-n3" 
]
# Set ALL_N to True if all input files for all N are already procesed
ALL_N = False

def processInputFiles(sub_dir_list):
    """ 
    Processes raw input files based on the specified --n_gram. Processed files are placed in specified 
    directory --proc_input_dir using the following rules:
        a) only words within length 5 and 15 are considered
        b) only alphanumeric words are considered after cleaning up: >|<|(|)|'|!|?|,|.|
       
       
    Ex: If file file_1 contains "Hello, good morning Putin!", then processed file <--proc_input_dir>\file_1 
    will contain "Hello morning" if '--n_gram 2'; "Hello\nmorning\nPutin" '--n_gram 1', 
    "Hello morning Putin" '--n_gram 3'.
        
    """
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

def getFeaturesN(lfile, feature_list, category, sub_dir, BINARY):
    """
    Used by constructTableAllN. This retrieves all processed counterparts (n=1/2/..) of a given file
    to perform filtering against the pool of n-grams from there. 
    
    Ex: given filename file_1, it will collect all words in <processed files dir for n=1>\<subdirectory>\file_1, 
    <processed files dir for n=2>\<subdirectory>\file_1, .. and then removes the words not in feature_list.
    """
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

def getFeatures(lfile, BINARY, return_count=False, feature_list=[]):
    """
    During IFP (return_count=False), this only extracts all words in lfile and returns the list.
    During SFC (return=True), this extracts all words in lfile, filter only those that are in the
    feature_list and returns frequency count (BINARY=False), or word presence (BINARY=True).
    
    """
    n_words = []
    with open(lfile) as f:
        n_words = f.read().splitlines()
    f.close()
   
    if return_count:
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
    """
    This collects all processed input files in the specified subdirectories.
    """
    for sub_dir in sub_dir_list:
        lfiles = os.listdir(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR, sub_dir))
        for lfile in lfiles:
            n_words += getFeatures(os.path.join(PROCESSED_INPUT_DIR, INPUT_DIR, sub_dir,lfile),BINARY)
    #unique-fy list 
    n_words = list(set(n_words))
    return n_words
    
def constructTableSubdir(feature_list, l_input_dir, l_label, sub_dir, lfiles, pri_index_table, PROCESSED_INPUT_DIR,BINARY,OUTPUT_DIR):
    """
    Called by constructTable and does the heavy lifting for each file in the given directory sub_dir.
    """
    for lfile in lfiles:
        print "Populating table for category %s, sub-directory %s, file %s " % (l_input_dir, sub_dir, lfile)
        sec_index_table = getFeatures(os.path.join(PROCESSED_INPUT_DIR, l_input_dir,sub_dir,lfile), BINARY, True, feature_list)
        pri_index_table = pd.concat([pri_index_table, sec_index_table], axis=1)
        
    pri_index_table.fillna(0, inplace=True)
    pri_index_table.columns = [range(0, pri_index_table.shape[1])]
    print "label: %s sub_dir: %s" % (l_label, sub_dir)
    print pri_index_table    
    constructSVMInputFiles(pri_index_table, l_label, sub_dir,OUTPUT_DIR)

    
def constructTable(feature_list, l_input_dir, l_label, sub_dir_list, PROCESSED_INPUT_DIR, BINARY,OUTPUT_DIR):
    """
    This constructs table with indexes=features/attributes using a single n-gram.
    Performs multithreaded table construction on multiple subdirectories based on the available core
    using constructTableSubdir.
    """
    pri_index_table = pd.DataFrame(index=feature_list)
    
    for sub_dir in sub_dir_list:
        lfiles = os.listdir(os.path.join(PROCESSED_INPUT_DIR, l_input_dir, sub_dir))
        constructTableSubdir_p = mp.Process(target=constructTableSubdir, args=(feature_list, l_input_dir, l_label, sub_dir, lfiles, pri_index_table,PROCESSED_INPUT_DIR,BINARY,OUTPUT_DIR))
        constructTableSubdir_p.start()
        
def constructTableAllN(feature_list, l_input_dir, l_label, sub_dir_list,PROCESSED_INPUT_DIR,BINARY,OUTPUT_DIR):
    """
    This constructs table with indexes=features/attributes using a pool of n-grams.
    This is NOT yet multithreaded.
    """
    pri_index_table = pd.DataFrame(index=feature_list)
    
    for sub_dir in sub_dir_list:
        for lfile in os.listdir(os.path.join(PROCESSED_INPUT_DIR,l_input_dir, sub_dir)):
            print "Populating table for category %s, sub-directory %s, file %s using N %d" % (l_input_dir, sub_dir, lfile, N_GRAM)

            sec_index_table = getFeaturesN(lfile, feature_list, l_input_dir, sub_dir,BINARY)
            pri_index_table = pd.concat([pri_index_table, sec_index_table], axis=1)
        
    pri_index_table.fillna(0, inplace=True)
    pri_index_table.columns = [range(0, pri_index_table.shape[1])]
    print "label: %s sub_dir: %s" % (l_label, sub_dir)
    print pri_index_table
    constructSVMInputFiles(pri_index_table, l_label, sub_dir,OUTPUT_DIR)

    
    
    
    
    
def constructSVMInputFiles(pri_index_table, l_label, sub_dir,OUTPUT_DIR):
    """
    This creates the actual SVM input files with the required format.
    """
    
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
    """
    Precursor to processInputFiles. Runs processInputFiles for each folder in each category.
    """
    
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
    """
    This only scans through all the processed files and collates all words, place them in file 
    <--proc_input_dir>\indexes_n_<--n_gram>.
    """
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
    """
    Precursor to constructTableAllN (if features used is a pool of n-grams)/ constructTable.
    This performs multithreaded SFC on all categories, depending on the core count.
    """
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
            constructTable_pn = mp.Process(target=constructTableAllN, args=(feature_list,INPUT_DIR, LABEL, sub_dir_list, PROCESSED_INPUT_DIR,BINARY,OUTPUT_DIR))
            constructTable_pn.start()
        else:
            constructTable_p = mp.Process(target=constructTable, args=(feature_list,INPUT_DIR, LABEL, sub_dir_list, PROCESSED_INPUT_DIR,BINARY,OUTPUT_DIR))
            constructTable_p.start()
        label_list+=1
        delay +=1 
        
def testargs():
    #print TESTING
    print BINARY
    print PROCESSED_INPUT_DIR
    #print PRE_PROCESS_INPUT
    #print N_GRAM
    #print GET_INDEX_MIN_FREQ
    #print GET_INDEX_TOP_FREQ
    #print MIN_WORD_FREQUENCY
    #print TOP_WORD_FREQUENCY
    print INDEX_FILE
    print OUTPUT_DIR

def getAllIndexes(cat_list, sub_dir_list):
    words = []
    for category in cat_list:
        for sub_dir in sub_dir_list:
            lfiles = glob.glob(os.path.join(PROCESSED_INPUT_DIR, category,sub_dir, "*"))
            for lfile in lfiles:
                with open(lfile) as f:
                    temp = f.read().splitlines()
                words+=temp
                f.close()
    return words
                    
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

    cat_list = ["cat1", "cat2"]
    sub_dir_list = ["1", "2", "3"]
    sub_dir_list = ["1"] 
    global TESTING
    global BINARY
    global PROCESSED_INPUT_DIR
    global PRE_PROCESS_INPUT
    global N_GRAM
    global GET_INDEX_MIN_FREQ
    global GET_INDEX_TOP_FREQ
    global MIN_WORD_FREQUENCY
    global TOP_WORD_FREQUENCY
    global INDEX_FILE
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description='MP')
    parser.add_argument('--proc_input_dir', dest='PROCESSED_INPUT_DIR', help='Location of the processed data files.',required=True)
    parser.add_argument('--binary', action='store_true', dest='BINARY', help='Binary output will be used.')
    parser.add_argument('--testing', action='store_true',  dest='TESTING', help='To process testing input files.')
    parser.add_argument('--pre_proc_input', action='store_true', dest='PRE_PROCESS_INPUT',   help='To process training input files.')
    parser.add_argument('--n_gram', dest='N_GRAM', help='N-gram to be used.',required=True)
    parser.add_argument('--get_index_min_freq', action='store_true',  dest='GET_INDEX_MIN_FREQ', help='To get indexes using minimum frequency of word occurrence.')
    parser.add_argument('--get_index_top_freq', action='store_true',  dest='GET_INDEX_TOP_FREQ', help='To get indexes using the top frequency of word occurrence.')
    parser.add_argument('--index_min_freq', dest='MIN_WORD_FREQUENCY', help='Minimum frequency of word occurrence to be used.')
    parser.add_argument('--index_top_freq', dest='TOP_WORD_FREQUENCY', help='Top frequency of word occurrence to be used.')
    parser.add_argument('--index_file', dest='INDEX_FILE', help='Index file to be used.')
    parser.add_argument('--output_dir', dest='OUTPUT_DIR', help='Location of the created SVM input files.',required=True)
   
    args = parser.parse_args()
    TESTING = args.TESTING
    BINARY = args.BINARY
    PROCESSED_INPUT_DIR = args.PROCESSED_INPUT_DIR
    PRE_PROCESS_INPUT = args.PRE_PROCESS_INPUT
    N_GRAM = int(args.N_GRAM)
    GET_INDEX_MIN_FREQ = args.GET_INDEX_MIN_FREQ
    GET_INDEX_TOP_FREQ = args.GET_INDEX_TOP_FREQ
    MIN_WORD_FREQUENCY = args.MIN_WORD_FREQUENCY
    TOP_WORD_FREQUENCY = args.TOP_WORD_FREQUENCY
    INDEX_FILE = args.INDEX_FILE
    OUTPUT_DIR = args.OUTPUT_DIR


    if not os.path.exists(PROCESSED_INPUT_DIR):
        os.mkdir(PROCESSED_INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

        
    if TESTING:
        # if --testing is used, then only subdirectory 4 go to IFP and SVC. 
        sub_dir_list = ["4"]
        preProcessInputFiles(cat_list, sub_dir_list)

    if PRE_PROCESS_INPUT:
        # if --pre_proc_input is used, then subdirectories 1,2,3 go to IFP. 
        preProcessInputFiles(cat_list, sub_dir_list)
        preGetFeatures(cat_list, sub_dir_list)
        return
        
    if GET_INDEX_MIN_FREQ:
        print "Getting indexes above minimum frequency.."
        words = getAllIndexes(cat_list, sub_dir_list)
        feature_counts = Counter(words)
        feature_list =  [word for word, count in feature_counts.items() if count > MIN_WORD_FREQUENCY]
        print "New feature vector length: %d" % (len(feature_list))
        f = open(os.path.join(INDEX_FILE), "w")
        f.writelines(["%s\n" % feature for feature in feature_list])
        f.close()
        return
    if GET_INDEX_TOP_FREQ:
        print "Getting indexes with top frequency.."
        words = getAllIndexes(cat_list, sub_dir_list)
        feature_counts = Counter(words)
        feature_counts = feature_counts.most_common()
        feature_list =  [word for word, count in feature_counts]
        feature_list = feature_list[:int(TOP_WORD_FREQUENCY)]
        print "New feature vector length: %d" % (len(feature_list))
        f = open(os.path.join(INDEX_FILE), "w")
        f.writelines(["%s\n" % feature for feature in feature_list])
        f.close()
        return
    
    #This part retrieves the features from file <--index_file>. 
    with open(INDEX_FILE) as f:
        feature_list = f.read().splitlines()
    time.sleep(10)
    print "Feature vector length: %d" % (len(feature_list))
    
    #This part starts the construction of SVM input files.
    preConstructTable(cat_list, feature_list, sub_dir_list)
     

if __name__ == '__main__':
    main()
    
    

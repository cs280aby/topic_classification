# PREREQUISITES:
1) Place this script in the same directory as the raw/unprocessed input files.
2) The raw/unprocessed input files should be in the following directories:
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
     
3) Training data/files should be in subdirectories 1,2,3. This is to support multiprocessing of input files.
4) Testing data/files should be in subdirectory 4. 

# STAGES/SAMPLE USAGE:
These should be performed in sequence:
1) input file processing (IFP) -- raw input files are cleaned and parsed according to the provided n-gram, see function processInputFiles for details.
Ex: python topic_classification_mod_2.py --proc_input_dir cat-input-dir  --pre_proc_input --n_gram 2 --output_dir cat-output-dir

-- this will 
2) index generation (IG) -- based on the provided n-gram, indexes are chosen based on the minimum/top frequency.
                            of word occurrence
Ex: python topic_classification_mod_2.py --proc_input_dir cat-input-dir  --pre_proc_input --n_gram 2 --output_dir cat-output-dir



3) svm file creation (SVC) -- based on the chosen indexes/features, SVM input files are created according to the 
                            chosen configuration (BINARY/COUNT)
                            
python topic_classification.py --proc_input_dir cat-input-dir --n_gram 2  --index_file cat-test-index --output_dir cat-output-dir --binary

python topic_classification.py --proc_input_dir cat_input_dir --n_gram 2  --index_file cat-test-index --output_dir cat_output_dir --binary --testing

                            
# NOTES:
- Terms indexes and features/attributes may be used interchangeably.


import glob
import os
from collections import Counter


categories = [
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
for category in categories:
    lfiles = glob.glob(os.path.join("processed-test", category, "*"))
    words = []
    for lfile in lfiles:
        with open(lfile) as f:
            temp = f.read().splitlines()
        words+=temp
        f.close()


count_words = Counter(words)
counts_words = counts.most_common()
f= open(os.path.join("processed-test", "indexes_count", "n_1_all" +".csv"), "w")
for key,value in count_words:
    f.write(str(key) + ", " + str(value) + "\n")
    #print "%s,%s" % (key, value)

f.close()




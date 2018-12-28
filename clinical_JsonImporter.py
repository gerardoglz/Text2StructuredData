import json
import pandas as pd
import os
import fnmatch
import numpy as np
import glob
import csv
from pprint import pprint

# Thousands of clinic letters have been pre-processed through NLP to extract clinical terms such as symptoms, diagnoses, etc.
# A patient may have one or more clinic letter(s) associated to them and each letter is an independent PDF file located
# in a secure server.
# The results of NLP preprocessing are saved in a single JSON file, which may or may not correspond to the directory
# structure of the PDF files in the server.
# The task is to relate the diverse data sources comprising of:
# 1)  the patient ids stored in a text file
# 2)  each PDF clinic letter in the server
# 3)  the NLP keywords for each letter within the JSON file
#
# NLP tool provides a list of keywords and phrases used in the clinical domain, containing assertions and denials of
# medical terms (e.g. patient has trouble sleeping, no issues when sleeping). We will use the 'mood' of the keyword to
# create a binary matrix there each patient has a positive or negative term associated to specific symptoms. For example,
#           sleepNormal
# patient1      0
# patient2      1
#
# The resulting matrix will be exported for a machine learning prediction model

outDir = '/media/gerardoglz/teradisk/clusterData/data/extracted_NIFTIS_Parkinsons/clinicalLettersResults'

# 1) import a csv file with a few thousands of patients with hospital IDs as their index, reformat them as necessary
df = pd.read_csv('/media/gerardoglz/brcii/E/a_imagepool_mr_II/x_ParkinsonsDisease/PDlist_1055cases/hospital_number_PD.txt', header=None)
df.columns = ['hospid']
df['hospid'] = df['hospid'].str.replace('/','-')

# 2) get the directory structure containing pdf files in the server
pdfFolder = '/media/gerardoglz/brcii/E/a_imagepool_mr_II/x_ParkinsonsDisease/PDlist_1055cases/x_CDR_reportPD1055-structured'
subdirName = pdfFolder.split('/')
datafilename = 'dfClinicalLettersDirsStructured_List-{}.h5'.format(subdirName[-1])
pdfdatafilename = 'dfClinicalLettersPDFsStructured_List-{}.h5'.format(subdirName[-1])
listing = glob.glob(os.path.join(outDir,datafilename))
listingPdfs = glob.glob(os.path.join(outDir,pdfdatafilename))
pathList = []
pdfList = []
if listing.__len__() > 0 and listingPdfs.__len__() > 0:
    # if we have done it previously, just load the listing
    dfPath = pd.read_hdf(os.path.join(outDir,datafilename), 'df_with_missing')
    dfPdfs = pd.read_hdf(os.path.join(outDir, pdfdatafilename), 'df_with_missing')
    print('loading data...')
else:
    # otherwise, go through the whole process
    for dirpath, dirs, files in os.walk(pdfFolder):
        for filename in fnmatch.filter(files, '*.pdf'):
            #re-format hospitalId if necessary
            hid = dirpath.split('/')
            pathList.append(hid[-1])
            pdfList.append(filename)
    # save subfolder names
    dfPath = pd.DataFrame({'pathlist':pathList})
    dfPath.to_hdf(os.path.join(outDir,datafilename), 'df_with_missing',format='table', mode='w')
    # save pdf filenames
    dfPdfs = pd.DataFrame({'pdfList': pdfList})
    dfPdfs.to_hdf(os.path.join(outDir, pdfdatafilename), 'df_with_missing', format='table', mode='w')
    print('saved data.')


#3) results of a pre-processing NLP stage has been saved as a JSON file, import the NLP keywords
jsonfile = '/media/gerardoglz/brcii/test_g/data/x_ParkinsonsDisease/clinicalLetters/pd_54k.json'
with open(jsonfile, 'r') as f:
    jsonparsed = f.readlines()


def ParseKeyTerms(data):
    #access to NLP clinical keywords within the JSON file

    #the structure of the NLP results are fixed, we could change this in the future
    k = 0
    dchList = data[u'Results'][k][u'ChunkingResponseApis'][0][u'ChunkingResult'][u'DetailedChunkList']
    termsList = [d['Term']  for d in dchList]
    keysList = [d['Key']  for d in dchList]
    uniqueterms = list(set(termsList))  # get unique NLP term entries (casting as set), optional to convert back to list: list(set(termsList))

    outList = []
    for i in range(len(uniqueterms)):
        keysText = []
        conceptText = []
        childrenText = []
        # find indices for each match, a unique term may be duplicate so get all corresp indices
        indexes = [k for k, j in enumerate(termsList) if j == uniqueterms[i]]

        # try to get sub-terms within children nodes (ChildConcepts and IMChildren), add them to list
        for ll in range(len(indexes)):
            keysText.append(keysList[indexes[ll]])

            childConceptList = dchList[indexes[ll]][u'ChildConcepts']
            for mm in range(len(childConceptList)):
                conceptText.append(childConceptList[mm][u'Term'])

            imChildList = dchList[indexes[ll]][u'IMChildren']
            for mm in range(len(imChildList)):
                childrenText.append(imChildList[mm][u'KeyText'])
        entry = [uniqueterms[i], keysText, conceptText, childrenText]
        outList.append(entry) #list of key terms per file
    return outList, uniqueterms


#get PDF filenames from JSON data
jsondataPdfnames = []
jsonPdfsfilename = 'dfPDFsWithinJsonStructured_List-{}.h5'.format(subdirName[-1])
listing = glob.glob(os.path.join(outDir,jsonPdfsfilename))
if(listing.__len__() > 0):
    dfJsonPdfs = pd.read_hdf(os.path.join(outDir,jsonPdfsfilename), 'df_with_missing')
    print('loading PDF filenames within Json list...')
else:
    ncols = len(jsonparsed)
    for j in range(ncols):
        tmp = json.loads(jsonparsed[j])
        fname = tmp[u'ExternalMessageId']
        jsondataPdfnames.append(fname)
    # save pdf filenames within Json
    dfJsonPdfs = pd.DataFrame({'jsondataPdfnames': jsondataPdfnames}).astype(str)
    dfJsonPdfs.to_hdf(os.path.join(outDir, jsonPdfsfilename), 'df_with_missing', format='table', mode='w')
    print('saved PDF filenames within Json list')

#use hospital ID as the main key that we will use to glue all data sources together
uniqueDf = pd.unique(df.hospid)

pdPatientsClinicalInfo = []
fullTermList = []
notFoundCases = []

for i in range(0,df.size):

    #get index of the pdf file in the directory list that matches current hospID
    idx = dfPath.loc[dfPath.pathlist == uniqueDf[i]]
    idxFound = idx.index.values

    if(len(idxFound)>0):
        nTerms = []
        pdfsInJsonShortlist = []
        # find the one with most information, then extract it
        for k in range(len(idxFound)):
            idx2load = idxFound[k]
            #find match between pdf file in folders and .pdf filename in json
            currPdfFilename = dfPdfs.pdfList[idx2load]
            idxPdfJson = dfJsonPdfs.loc[dfJsonPdfs.jsondataPdfnames == currPdfFilename]
            idxPdfJsonFound = idxPdfJson.index.values

            if (len(idxPdfJsonFound) > 0):
                datatmp = json.loads(jsonparsed[idxPdfJsonFound[0]]) #always use first index if more than 1

                #check how many key words have been extracted
                tstres = datatmp[u'Results']
                if(tstres is not None):
                    ml = len(datatmp[u'Results'][0][u'ChunkingResponseApis'][0][u'ChunkingResult'][u'DetailedChunkList'])
                    nTerms.append(ml)
                    pdfsInJsonShortlist.append(idxPdfJsonFound[0]) #save index of json entry
                else:
                    nTerms.append(0)
                    pdfsInJsonShortlist.append(0)

        #of all pdf letters extracted, we will only focus on the one with most keywords extracted
        idxDateMatched = []
        idxMatch = 0
        if(len(nTerms)>0):
            idxMax = nTerms.index(max(nTerms))
            idxDateMatched = idxMax
            idxMatch = 1

        #extract and parse the actual keywords
        pdffilename = ''
        if(idxMatch == 1):
            jsonMatch = json.loads(jsonparsed[pdfsInJsonShortlist[idxMax]])
            pdffilename = jsonMatch[u'ExternalMessageId']
            [resList, uniqueTerms] = ParseKeyTerms(jsonMatch)

        entry = [uniqueDf[i], pdffilename, resList]
        pdPatientsClinicalInfo.append(entry)

        for f in range(len(uniqueTerms)):
            fullTermList.append(uniqueTerms[f])
    else:
        print('HospID Not found: ', uniqueDf[i])
        notFoundCases.append(uniqueDf[i])


print( "Number of cases not found: {}".format(len(notFoundCases)))
#we will save the list of cases not found
casesname2save = "pdfCasesNotFound.csv"
dfNotFoundCases = pd.DataFrame(data=notFoundCases)
dfNotFoundCases.to_csv(os.path.join(outDir,casesname2save),encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC, header=False, index=False)

# clinical keywords based on the NLP tool we used may be duplicated due to the design of the NLP report
# clean up keyword list into unique list
uniquefullTermList = pd.unique(fullTermList)
dfUniquefullTermList = pd.DataFrame({'termslist':uniquefullTermList})
dfUniquefullTermList = dfUniquefullTermList.sort_values(by=['termslist'], inplace=False)
dfUniquefullTermList = dfUniquefullTermList.reset_index(drop=True)
binFullClinicalTerms = np.zeros([len(pdPatientsClinicalInfo),len(uniquefullTermList)])
binFullClinicalTerms.fill(np.nan)

#categorise the patient as a healthy subject (noPresence or denials) or subject with a
# diagnosis (having any abnormality or assertions)
noPresence = ['known absent', 'not present', 'not current', 'normal']
abnormalities = ['known possible','low','moderate','small','increase','recent',
        'multiple','current','abnormal','slow','left','right','in progress',
        'suspected']

# create a binary array for keywords found.
#for each keyword, find whether there is a description about it such as an adjective and compare to our own
# mini-dictionary for denials and assertions, more focused into trying to find a meaning when there is an issue with the
# keyword (e.g. 'sleep' and 'not present' = 0 ("patient doesn't have problems with sleeping", rather than "patient doesn't sleep"))
indexColumn = []
for p in range(len(pdPatientsClinicalInfo)):
    # for each patient, get hospital id
    indexColumn.append(pdPatientsClinicalInfo[p][0])
    keywords = []
    res = []

    for kr in range(len(pdPatientsClinicalInfo[p][2])):
        #get children for current keyword to get adjectives and descriptions
        pos = 0
        kword = pdPatientsClinicalInfo[p][2][kr][0]
        ChildConcepts = pdPatientsClinicalInfo[p][2][kr][2]
        IMChildren = pdPatientsClinicalInfo[p][2][kr][3]
        if (len(ChildConcepts) == 0 and len(IMChildren) == 0):
            #if empty, take for granted it is a positive based on the term (patient has a problem with the current description)
            pos = 1
        elif len(ChildConcepts) > 0:
            #match concepts with our positive and negative dictionaries

            #compare if there is any negative
            matching = [s for s in ChildConcepts if any(xs.lower() in s.lower() for xs in noPresence)]
            if(len(matching) > 0):
                #we have a negative
                pos = 0

            #positives have precedence, so compare for positives even if there was a negative
            matchingPos = [s for s in ChildConcepts if any(xs.lower() in s.lower() for xs in abnormalities)]
            if (len(matchingPos) > 0):
                pos = 1
        elif len(IMChildren) > 0:
            #if no matching concepts but there is any description in children, assume it is a positive
            pos = 1
        keywords.append(kword)
        res.append(pos)

    for t in range(len(keywords)):
        #check the positive and negative entries and assign them to the correct index in the binary matrix
        idx = dfUniquefullTermList.loc[dfUniquefullTermList.termslist == keywords[t]]
        idxFound = idx.index.values
        #set those indices as a positive/negative entry
        for pp in range(len(idxFound)):  # idxFound
            binFullClinicalTerms[p, idxFound[pp]] = res[t]

# save binary array for ML model
name2savedData = "binFullClinicalTerms{}x{}c".format(len(pdPatientsClinicalInfo),len(uniquefullTermList))
np.save(os.path.join(outDir,name2savedData), binFullClinicalTerms)

print('finished')

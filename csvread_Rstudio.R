# 'Impromptu' code to import a .csv file exported from a database containing clinical data with longitudinal cases (hospital visit dates) for a cohort of patients.
# A patient may have diverse MRI scans during one or more visits, we aim to record only those scans with a special 'contrast agent' (similar to angiography) acquired within a period of time.

# The aim of this code is to compare the performance of a simple algorithm against other implementations written in Matlab and Python. Parts of the code have been deleted due to the sensitivity of the data.

# Data source consists of:
# subjectId: patient ID
# date: date of visit(s) to hospital
# scan_is_contrast_enhanced: whether a patient has been scanned with an special MRI scan

# Note: Standard R assign syntax (<-) has been replaced by = for clarity 

tic = Sys.time()
 
pathtofile = 'AnonymisedDataGoesHere.csv'
csvfile = read.csv(pathtofile, header = TRUE, sep = ';')

periodThreshold = 60 #number of days 

currDateEnhanced = list()
currSortedDates = list()
bContrastChange = list()

#We will sort the data based on subject ids, hence we eachunique subjectId as a key to explore our database
uSubject = unique(csvfile[1])

for(i in 1:nrow(uSubject)) { 
  
  #for each subject, let's find the corresponding hospital visit dates   
  currSubj = uSubject$xnat_subjectid[i] 
  idxSubjects = which(csvfile$xnat_subjectid == currSubj) #get the index entries for each subject
  currDates = as.Date(csvfile$date[idxSubjects]) # get visit dates for each subject
  uDates = unique(currDates) #sort visit dates from oldest to latest (2+ visits on the same day are assumed to be a single visit)
  uDates = sort(uDates)
  
  #keep a vector containing if a special MRI scan with 'contrast agent' (similar to angiography) was aqcuired in the dates
  contrastDay = numeric(length(uDates)) 
  for(j in 1:length(uDates)) { 
  
    # for each date, get the corresponding indices in our database
    idxUDates = which(currDates == uDates[j]) 
	
	# was there a 'contrast' MRI scan acquired on these days ?
    if(any(csvfile$scan_is_contrast_enhanced[idxSubjects[idxUDates]] == 1)) {
      contrastDay[j] = 1
    } else { 
      contrastDay[j] = 0 
    }
    
	#only keep track of the hospital visits within the last N number of days
    if(j>1) {
      if(uDates[j] - uDates[j-1] < periodThreshold && contrastDay[j] == 1 && contrastDay[j-1] == 0) {
        bContrastChange[i] = 1
        currSortedDates[[i]] = uDates
      }
    }    
  }

  currDateEnhanced[[i]] = contrastDay  
  
  #other processing steps have been deleted here due to the sensitivity of the data
}

# how long did it take to process all the entries
toc = Sys.time()
print(toc - tic)

#arrange the results as lists so we can export the data at a later time (if needed)
numContrastMatching = length(which(lapply(bContrastChange, is.null) == FALSE)) # count
idXPrunedList = (which(lapply(bContrastChange, is.null) == FALSE))
patientList = as.list(as.character(csvfile$xnat_experimentid[idXPrunedList]))
patientContrastList = currDateEnhanced[idXPrunedList]
patientDatesList = currSortedDates[idXPrunedList]

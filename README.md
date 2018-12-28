# Text2StructuredData

Assorted code used to import text data into a processing pipeline:

*csvread_Rstudio.R* - R code to import a .csv file exported from a database containing clinical data with longitudinal cases (hospital visit dates) for a cohort of patients. A patient may have diverse MRI scans during one or more visits, we aim to record only those scans with a special 'contrast agent' (similar to angiography) acquired within a period of time.
The aim of this code is to compare the performance of a simple algorithm against other implementations written in Matlab and Python. Parts of the code have been deleted due to the sensitivity of the data.

 Data source consists of:
 1) subjectId: patient ID
 2) date: date of visit(s) to hospital
 3) scan_is_contrast_enhanced: whether a patient has been scanned with an special MRI scan

Note: Standard R assign syntax (<-) has been replaced by = for clarity 


*clinical_JsonImporter.py* - Python code to read thousands of clinic letters that have been pre-processed through NLP to extract clinical terms such as symptoms, diagnoses, etc. A patient may have one or more clinic letter(s) associated to them and each letter is an independent PDF file located in a secure server. The results of NLP preprocessing are saved in a single JSON file, which may or may not correspond to the directory structure of the PDF files in the server.
 The task is to relate the diverse data sources comprising of:
 1)  the patient ids stored in a text file
 2)  each PDF clinic letter in the server
 3)  the NLP keywords for each letter within the JSON file

 NLP tool provides a list of keywords and phrases used in the clinical domain, containing assertions and denials of  medical terms (e.g. patient has trouble sleeping, no issues when sleeping). We will use the 'mood' of the keyword to create a binary matrix there each patient has a positive or negative term associated to specific symptoms. For example,
           sleepNormal
 patient1      0
 patient2      1

 The resulting matrix will be exported for a machine learning prediction model

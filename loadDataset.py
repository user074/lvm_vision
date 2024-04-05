import json
import sys
import os
import urllib.request as ureq
import requests

import pdb

download=1 # 0 if images are already downloaded

###############################################################
######################### load dataset json file ###############
################################################################
with open('dataset.json', 'r') as fp:
        data = json.load(fp)

## dictionary data contains image URL, questions and answers ##




################################################################
############### Script for downloading images ##################
################################################################
## Make a directory images to store all images there ##########
if download == 1:
    os.makedirs('./images', exist_ok=True)
    for k in data.keys():
        ext = os.path.splitext(data[k]['imageURL'])[1]
        outputFile = 'images/%s%s' % (k, ext)
        
        if os.path.exists(outputFile):
            print(f"Skipping {outputFile} as it already exists.")
        else:
            print('Downloading %s to %s' % (data[k]['imageURL'], outputFile))
            try:
                response = requests.get(data[k]['imageURL'])
                if response.status_code == 200:
                    with open(outputFile, 'wb') as file:
                        file.write(response.content)
                else:
                    print(f"Failed to download image: {data[k]['imageURL']}")
            except requests.exceptions.ConnectionError as e:
                print(f"Error downloading image: {data[k]['imageURL']}")
                print(f"Error message: {str(e)}")
                # Handle the error or skip to the next URL

#################################################################
################### Example of data access #####################
################################################################
for k in data.keys():
    ext=os.path.splitext(data[k]['imageURL'])[1]
    imageFile='images/%s%s'%(k,ext)

    print('************************')
    print('Image file: %s'%(imageFile))
    print('List of questions:')
    print(data[k]['questions'])
    print('List of corresponding answers:')
    print(data[k]['answers'])
    print('Use this image as training (1), validation (2) or testing (3): %s'%(data[k]['split']))
    print('*************************')





######################################################################
########################### Get dataset stats ########################
######################################################################
genSet=set()
for k in data.keys():
    genSet.add(data[k]['genre'])



numImages=len(data.keys())
numQApairs=0
numWordsInQuestions=0
numWordsInAnswers=0
numQuestionsPerImage=0
ANS=set() # Set of unique answers
authorSet=set()
bookSet=set()


for imgId in data.keys():
    numQApairs = numQApairs+len(data[imgId]['questions'])
    numQuestionsPerImage = numQuestionsPerImage + len(data[imgId]['questions'])
    authorSet.add(data[imgId]['authorName'])
    bookSet.add(data[imgId]['title'])

    for qno in range(len(data[imgId]['questions'])):
        ques=data[imgId]['questions'][qno]
        numWordsInQuestions = numWordsInQuestions+len(ques.split())
    for ano in range(len(data[imgId]['answers'])):
        ans=data[imgId]['answers'][ano]
        ANS.add(ans)
        numWordsInAnswers = numWordsInAnswers+len(str(ans).split())



print("--------------------------------")
print("Number of Images: %d" %(numImages))
print("Number of QA pairs: %d" %(numQApairs))
print("Number of unique author: %d" %(len(authorSet)))
print("Number of unique title: %d" %(len(bookSet)))
print("Number of unique answers: %d" %(len(ANS)))
print("Number of unique genre: %d" %(len(genSet)))
print("Average question length (in words): %.2f" %(float(numWordsInQuestions)/float(numQApairs)))
print("Average answer length (in words): %.2f" %(float(numWordsInAnswers)/float(numQApairs)))
print("Average number of questions per image: %.2f" %(float(numQuestionsPerImage)/float(numImages)))
print("--------------------------------")


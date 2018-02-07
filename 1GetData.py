import praw
import auth_info
import time
import nltk
import os
import pandas as pd
import datetime
import pickle
import string
from newspaper import Article

#####Initilaization Parameters#####
startnewdf=False
csvpath=os.getcwd()+'/data/postdata.csv'
excludesites=['redd.it','youtube.com','imgur']
subredditname=['conservative','liberal','democrats','republican','politics','socialism','libertarian','the_congress']
sortby=['top','hot','controversial','new']
subids=list()
#####Initilaization Parameters#####
def bot_login():
	print('Logging in...')
	r=praw.Reddit(username=auth_info.username,
		password=auth_info.password,
		client_id=auth_info.client_id,
		client_secret=auth_info.client_secret,
		user_agent='hoteinokodomo tests reddit bot')
	print('Logged in!')
	return r

def clean_text(inputstring):
	cleanstring=(inputstring.translate(str.maketrans('','',string.punctuation))).lower()
	clearnstring=' '.join(cleanstring.split())
	return cleanstring

def run_bot(r,subred,sorttype):
	if startnewdf:
		commentFrame=pd.DataFrame(columns=['subreddit','title','url','author','upvotes','downvotes','sortedby','whenposted','websitetext','replies','submissionID'])
		for i in commentFrame:
			commentFrame[i]=list()
	else:
		commentFrame=pickle.load(open(os.getcwd()+'/data/postdata.pickle','rb'))
		subids=list(commentFrame.submissionID)

	if sorttype=='hot':
		allsubmissions=r.subreddit(subred).hot(limit=1000)
	elif sorttype=='controversial':
		allsubmissions=r.subreddit(subred).controversial(limit=1000)
	elif sorttype=='new':
		allsubmissions=r.subreddit(subred).new(limit=1000)
	elif sorttype=='top':
		allsubmissions=r.subreddit(subred).top(limit=1000)

	for submission in allsubmissions:#for submission in r.subreddit('conservative').controversial('day'):
		if not any(url in submission.url for url in excludesites):
			if submission.id not in subids:
				subids.append(submission.id)

				try:
					if ('reddit' in submission.url):
						html2text=submission.selftext
					else:
						html = Article(submission.url)
						html.download()
						html.parse()
						html2text=clean_text(html.text)
				except:
					html2text=float('NaN')

				usercomments=dict()
				for toplevelcomm in submission.comments:
					if hasattr(toplevelcomm,'author'):#some comment results returned by PRAW don't have associated author, leave them out
						if toplevelcomm.author not in usercomments.keys():
							usercomments[toplevelcomm.author]=list()
						try:
							usercomments[toplevelcomm.author].append(toplevelcomm.body)
						except:
							usercomments[toplevelcomm.author].append(float('NaN'))

				commentFrame.loc[len(commentFrame)]=[subred,submission.title,submission.url,submission.author,submission.ups,submission.downs,sorttype,datetime.datetime.fromtimestamp(submission.created),html2text,usercomments,submission.id]

				print(datetime.datetime.fromtimestamp(submission.created))
			else:
				alreadyexists+=1
	
	if os.path.isfile(csvpath):
		with open(csvpath,'a') as f:
			commentFrame.to_csv(f,header=False)

	else:
		commentFrame.to_csv(csvpath)
	
	pickle.dump(commentFrame,open(os.getcwd()+'/data/postdata.pickle','wb'))

r=bot_login()

for sub in subredditname:
	for sortsub in sortby:
		run_bot(r,sub,sortsub)

import pandas as pd #importing libraries
import gzip 
from afinn import Afinn
import statistics
import matplotlib.pyplot as plt
afinn = Afinn(language='en')

def parse(path): #lines 9-20 were from the UCSD Amazon Project Data website as it gave us code on how to unzip the data and extract it to a dataframe for easy use
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('meta_Office_Products.json.gz') #we unzip both of our json files into two separate dataframes
df2 = getDF('reviews_Office_Products.json.gz')

data = df2.loc[25,:].to_dict() #when we selected our user, we needed a way to extract her data so this line of code extracts it for us
reviewer_data = [[key, value] for key, value in data.items()] #it appends her data from line 25

def find_product(df, asin_to_find): #this procedure will print the title and description of the product that has the asin value that we give it
    product = df.loc[df['asin'] == asin_to_find]
    if product.empty:
        print(f"No product found with ASIN {asin_to_find}.")
    else:
        print("Title:", product['title'].values[0])
        print()
        print("Description:", product['description'].values[0])
        print()

find_product(df, reviewer_data[1][1]) #it will look in the first dataframe and use the user's asin value to find the product title and description

#sentiment analysis
print('Review: ' + reviewer_data[4][1]) #we print the review of the user
print('Sentiment Score: ' + str(afinn.score(reviewer_data[4][1]))) #we print the sentiment score of the user after running it through AFINN
print()

review_list = [] #setting up an empty list

for index, row in df2.iterrows(): #this checks for any products that have the same asin value as our user and appends their review to the list
    if row["asin"] == reviewer_data[1][1]:
        review_list.append(row["reviewText"])

sentiment_scores = [] #setting up an empty list

for sentence in review_list: #for every sentence in that list it will run it through AFINN
  sentiment_scores.append(afinn.score(sentence))

mode = statistics.mode(sentiment_scores)
print('Scores of other reviewers(mode): ' + str(mode)) #this will print out the mode of the scores of the reviews. it gives us a general idea of what most people thought of the product
print()

asin_value = reviewer_data[1][1] #we set our user's asin value to a variable so we can use it later on
related_values = []

filtered_df = df[df['asin'] == asin_value] #this filtered dataframe will only include data with the same asin value as our user's

related_values.extend(filtered_df['related'].apply(lambda x: x).tolist()) #this adds the data from 'related' to the list related_values. this list has a list of all the products users have looked at or bought that have also bought the calendar. 

also_viewed = related_values[0]['also_viewed'] #we create a separate list but this time we only keep the also_viewed products

review_texts = {} #setting up a dictionary

for asin in also_viewed: #for every asin value in the list also_viewed, it will find products that have the same asin value, and append the reviewText value to the dictionary
  matching_rows = df2[df2["asin"] == asin]
  review_texts[asin] = []
  for index, row in matching_rows.iterrows():
    review_texts[asin].append(row["reviewText"])

remove_list = [] #setting up a list. this list stores asin values that need to be removed. there are some asin values that do not have a review text. 

for asin, texts in review_texts.items(): #this checks in the dictionary above to see if there are any asin values that have no reviews and appends them to the remove list. 
  if len(texts) == 0:
    remove_list.append(asin)

for asin in remove_list: #this will take the asins from the remove list and remove them from our review_texts list. 
    del review_texts[asin]

for asin, texts in review_texts.items():
    #create an empty list to store the AFINN scores for each sentence
    scores = []
    #loop through the review texts for this asin
    for text in texts:
        #split the text into sentences
        sentences = text.split(".")
        #loop through the sentences and calculate the AFINN score
        for sentence in sentences:
            score = afinn.score(sentence)
            scores.append(score)
    #replace the review text value with the AFINN score
    review_texts[asin] = scores

for asin, scores in review_texts.items(): #this will find the average AFINN score for every asin value
    review_texts[asin] = sum(scores) / len(scores)

highest_scores = [] #setting up an empty list

highest_score = max(review_texts, key=review_texts.get) #lines 110-111 will find the highest AFINN score and append the title and score of that product to the list highest_scores
highest_scores.append((highest_score,review_texts[highest_score]))

titles = [df.loc[df['asin'] == asin, 'title'].values[0] for asin, score in highest_scores] #this takes the asin value in the list above and looks through the dataframe for its matching asin value and saves its title to the list. 

print('Product Our User Would Most Likely Buy: ' + str(titles[0])) #this is where we suggest which product our user is most likely to buy
print()

for asin, score in review_texts.items(): #there are lots of AFINN values with trailing numbers so we round them to make it easier to view on our bar chart
    #round the score to the tenth place
    rounded_score = round(score, 1)
    #replace the score with the rounded score
    review_texts[asin] = rounded_score
  
asins = list(review_texts.keys()) #we set the asins to the keys
scores = list(review_texts.values()) #we set the scores to the values

plt.barh(asins, scores) #asins is the y values and the scores are the x values
plt.ylabel("Product")
plt.xlabel("AFINN Score")
plt.title("AFINN Scores for each Product(ASIN)")
plt.yticks(fontsize=5, rotation=0, va='center') #we make the font size for the y ticks to be 5 otherwise it'll overlap each other and will make it hard to read
plt.show()
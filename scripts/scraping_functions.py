#!/usr/bin/env python
# coding: utf-8

# # Modules <a class="anchor" id="chapter1"></a>

# In[22]:


import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import os
from datetime import date


# # API calls <a class="anchor" id="chapter3"></a>

# We'll now write a function to request the data from TMDB. The function takes a vector of film IDs as input and it returns a dataframe. For each ID it sends a GET request to the API. If the status code is equal to 200 (i.e. the request has been successful) it appends the data to a Pandas dataframe. Otherwise, it just appends the film_id and fills the remaining columns with null values.

# In[2]:


def build_film_df(film_ids):
    
  api_key = os.environ.get("tmdb_api_key")

  df = pd.DataFrame(columns = ["id", "title", "release_date", "runtime", "country", "language", 
                               "genre", "studios", "budget", "revenue"])

  for film_id in film_ids:
      
      response = requests.get(f"https://api.themoviedb.org/3/movie/{film_id}?api_key={api_key}")

      if response.status_code == 200:  

        response_json = response.json()

        df = df.append({"id": response_json["imdb_id"],
                        "title": response_json["title"],
                        "release_date": response_json["release_date"],
                        "runtime": response_json["runtime"],
                        "country": ';'.join([country['name'] for country in response_json["production_countries"]]),
                        "language": ';'.join([language["english_name"] for language in response_json["spoken_languages"]]),
                        "genre": ';'.join([genre["name"] for genre in response_json["genres"]]),
                        "studios": ';'.join([company["name"] for company in response_json['production_companies']]),
                        "budget": response_json['budget'],
                        "revenue": response_json["revenue"]}, 
                        ignore_index = True)
        
      else:
        df = df.append({"id": film_id}, ignore_index = True)

  return df


# # Scraping additional data <a class="anchor" id="chapter4"></a>

# TMDB API has some limits: it doesn't provide data about the people who worked on a film (directors, writer, actors etc.) and its rating data is inaccurate. We'll thus integrate the data we just got by scraping some more information.

# ## Custom functions <a class="anchor" id="subparagraph1"></a>

# To implement sound software engineering principles we'll scrape the data by building a function for each type of data we need.

# ### Film ID

# To scrape the film ID, we'll first access the 'meta' tag with property equal to 'imdb:pageConst' and then we'll get the value of the 'content' attribute

# In[3]:


def scrape_film_id(soup):
    
    try:
        film_id = soup.find("meta", {"property": "imdb:pageConst"}).get("content")
    except:
        return np.nan
    else:
        return film_id


# ### Directors

# To scrape the directors we'll access all the href tags containing the 'tt_ov_dr' regex and with classes equal to "ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link". The resulting list contains duplicated values, so we'll turn it into a set to keep only distinct values. We'll finally turn it back into a list which we'll collapse into a single string.

# In[4]:


def scrape_director(soup):
    
    try:  
        a_tags = soup.find_all(href = re.compile("tt_ov_dr"), class_="ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link")
    
        directors = list(set([a.text for a in a_tags]))
    
        directors = ';'.join(directors)
        
    except:
        return np.nan
        
    else:
        return directors


# ### Writers

# Same as before but this time we're looking for href tags containing the 'tt_ov_wr' regex.

# In[5]:


def scrape_writer(soup):
    
    try:
        a_tags = soup.find_all(href = re.compile("tt_ov_wr"), class_="ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link")
        
        writers = list(set([a.text for a in a_tags]))
        
        writers = ';'.join(writers)
        
    except:
        return np.nan
        
    else:
        return writers


# ### IMDB average rating

# The rating can be found as the text of the first span tag with class equal to 'AggregateRatingButton__RatingScore-sc-1ll29m0-1 iTLWoV'

# In[6]:


def scrape_imdb_rating(soup):
    
    try:   
        span = soup.find_all("span", class_="AggregateRatingButton__RatingScore-sc-1ll29m0-1 iTLWoV")[0]
    
        imdb_rating = span.text
        
    except:
        return np.nan
    
    else:
        return imdb_rating        


# ### IMDB rating count

# To scrape the rating count we need to get a little creative. This value can be found in the 'script' tag with type equal to 'application/ld+json': it's preceded by '"ratingCount":'. If we treat the whole tag as string, we can extract it with a positive lookbehind.

# In[7]:


def scrape_rating_count(soup):
    
    try:
        pattern = r'(?<="ratingCount":)[\d.]+'
        string = str(soup.find("script", {"type": "application/ld+json"}))
        rating_count = re.findall(pattern = pattern, string = string)[0]
        
    except:
        return np.nan
    
    else:
        return rating_count


# ### Metascore

# The 'span' tag with the 'score-meta' class contains the Metascore of the film

# In[8]:


def scrape_metascore(soup):
    
    try:
        metascore = soup.find("span", class_="score-meta").text 
    
    except:
        return np.nan
    
    else:
        return metascore


# ### User review count

# We'll extract the number of user reviews by treating the 'script' tag with id equal to '__NEXT_DATA__' as a string and by using a regex with both a lookbehind and a lookahead.

# In[9]:


def scrape_user_review_count(soup):
    
    try: 
        pattern = r'(?<="total":)\d+(?=,"__typename":"ReviewsConnection"},"criticReviewsTotal":)'
  
        string = str(soup.find("script", {'id': '__NEXT_DATA__'}))
    
        user_review_count = re.findall(pattern = pattern, string = string)[0]
        
    except:
        return np.nan
    
    else:
        return user_review_count


# ### Critic review count

# To retrieve the number of critic reviews we'll first access all the 'span' tags assigned to a class containing the 'three-Elements' regex and then we'll filter the resulting list by only keeping the 'span' tag containing the word 'Critic'. Finally, we'll use a very simple regex to extract the count.

# In[10]:


def scrape_critic_review_count(soup):
    
    try:   
        spans = soup.find_all("span", class_= re.compile("three-Elements")) 
        
        string = list(filter(lambda x: 'Critic' in str(x), spans))[0].text
        
        critic_review_count = re.findall(r'\d+', string)[0]
    
    except:
        return np.nan
        
    else:
        return critic_review_count


# ### Color

# The information about whether the film is in colour or in black and white sits in the href attribute with the word 'color' in it and with classes equal to 'ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link'

# In[11]:


def scrape_color(soup):
    
    try:
        a_tag = soup.find_all(href = re.compile('colors'), class_="ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link")[0]
        
        color = a_tag.text
        
    except:
        return np.nan
    
    else:
        return color


# ### Aspect ratio

# The aspect ratio will be scraped similarly by using a positive lookbehind.

# In[12]:


def scrape_aspect_ratio(soup):
    try:
        pattern = r'(?<="aspectRatio":")[\d.\s:]+'
    
        string = str(soup.find("script", {"type": "application/json"}))
    
        aspect_ratio = re.findall(pattern = pattern, string = string)[0] 
        
    except:
        return np.nan
        
    else:
        return aspect_ratio


# ## Creating the dataframe <a class="anchor" id="subparagraph2"></a>

# We'll create a dataframe of scraped data by iterating over a list of URLs and by appending each scraped value to a column of the df. Should we get an error while scraping, we're ll retry up to 3 times by waiting approximately 9 minutes between each try.

# In[13]:


def build_scraped_df(urls):

  df = pd.DataFrame(columns = ["id", "director", "writer", "imdb_rating", "imdb_rating_count", "metascore", 
                               "user_review_count", "critic_review_count", "color", "aspect_ratio", "last_updated"])

  for url in urls:
        
        session = requests.Session()
        retry = Retry(total = 30, backoff_factor = 0.000001)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        content = session.get(url).content
        soup = BeautifulSoup(content)
        
        df = df.append({"id": scrape_film_id(soup),
                        "director": scrape_director(soup),
                        "writer": scrape_writer(soup),
                        "imdb_rating": scrape_imdb_rating(soup),
                        "imdb_rating_count": scrape_rating_count(soup),
                        "metascore": scrape_metascore(soup),
                        "user_review_count": scrape_user_review_count(soup),
                        "critic_review_count": scrape_critic_review_count(soup),
                        "color": scrape_color(soup),
                        "aspect_ratio": scrape_aspect_ratio(soup),
                        "last_updated": date.today()}, 
                        ignore_index = True)



  return df


# # Scraping people data <a class="anchor" id="chapter5"></a>

# ## Custom functions <a class="anchor" id="subparagraph3"></a>

# ### Full cast

# The names of the artists can be found inside the table with the "cast-list" class: they're the alternative text of the images inside the table.

# In[14]:


def scrape_actors(soup):
    
    try:   
        images = soup.find("table", class_="cast_list").find_all("img") 
        
        actors = [img.get("alt") for img in images]  
        
        actors = ';'.join(actors)
    
    except:
        return np.nan
    
    else:
        return actors


# ### Cinematographer, editor, composer, producers, production designer, art director, costume designer

# The following function scrapes the names of different types of artists. The type must be specified in the 'artist' argument. 
# 
# In the "Full Cast & Crew" pages each artist type has its own table element containing the names of the artists. Each of these tables is preceded by an h4 element with an id equal to the artist type. For example, the name of the cinematographer is contained in the table preceded by the h4 element with id equal to "cinematographer". To scrape this data, we'll first access the h4 element, then we'll access the table next to it and finally we'll get all the names.

# In[15]:


def scrape_artist(soup, artist):
    
    try:
        h = soup.find("h4", id = artist)
        
        a_tags = h.find_next("table").find_all("a")
        
        artists = [a.text.lstrip().replace('\n', '') for a in a_tags]
        
        artists = ';'.join(artists)
    
    except:
        return np.nan
        
    else:     
        return artists


# ## Creating the dataframe <a class="anchor" id="subparagraph4"></a>

# To build the second dataframe of scraped data, we'll create a similar function to the one we used before.

# In[16]:


def build_scraped_df_2(urls):
    
    df = pd.DataFrame(columns = ["id", "actors", "cinematographer", "editor", "composer", "producers", "production_designer",
                                "art_director", "costume_designer"])
    
    for url in urls:
        session = requests.Session()
        retry = Retry(total = 30, backoff_factor = 0.000001)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        content = session.get(url).content
        soup = BeautifulSoup(content)
            
        df = df.append({"id": re.findall(r"tt\d+", url)[0],
                        "actors": scrape_actors(soup),
                        "cinematographer": scrape_artist(soup, "cinematographer"),
                        "editor": scrape_artist(soup, "editor"),
                        "composer": scrape_artist(soup, "composer"),
                        "producers": scrape_artist(soup, "producer"),
                        "production_designer": scrape_artist(soup, "production_designer"),
                        "art_director": scrape_artist(soup, "art_director"),
                        "costume_designer": scrape_artist(soup, "costume_designer")},
                        ignore_index = True)
        
    return df


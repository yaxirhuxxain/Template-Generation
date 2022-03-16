# -*- coding: utf-8 -*-

from github3 import login
from getpass import getpass
import time
import os

def git_login(user):
    password = getpass('GitHub password for {0}: '.format(user))
    g = login(user, password)
    return g


def git_cloner(stars, query, language, numRepos, g):
    out_dir  = "./Repos_info"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    searchQuery = '{}stars:{} language:{}'.format(str(query), str(stars), str(language))
    SearchResult = g.search_repositories(searchQuery, sort="stars", number=numRepos)
    count = 0
    
    for result in SearchResult:
        name = result.repository.name
        clone_url = result.repository.clone_url
        num_stars = result.repository.stargazers
        
        print("{}>>{}>>{}>>{}".format(count,num_stars,name,clone_url)) 
        time.sleep(0.5)
        count +=1
        
        with open(os.path.join(out_dir, language+'-clone_url.txt'), 'a+') as f:
            f.write("({},{},{},{})".format(count,num_stars,name,clone_url))
            f.write("\n")
            
    
            
    print("Total number of repos found: "+str(count))
    print("Clone list generation done....")
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

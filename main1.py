import emotion1
#import interface
from scraper import locate_url, rank_movies, scrape_IMDB, scrape_rt
import movie_page
from movie_summary import get_movie_summary
from similarity_analyzer1 import find3MostSim
import summary_page

from tkinter import *
#import emotion1
if __name__ == "__main__":

   # # call interface.py

    #window = Tk()
    #interface = interface.interface(window)
    #window.mainloop()
    #user_inputs = [] # obtain user selections
    #for i in interface.result:
     #   user_inputs.append(interface.emotions[i])

    # apply self-built crawler

    user_emotion = emotion1.final_output
    url_lst = locate_url(user_emotion)
    movie_dict = {}

    for url in url_lst:
        if "www.imdb.com" in url:
            if (user_emotion) == "happy" :
                movie_dict.update(scrape_IMDB(url, 20))
            elif (user_emotion) == "sad":
                movie_dict.update(scrape_IMDB(url, 20))
            elif (user_emotion) == "neutral":
                movie_dict.update(scrape_IMDB(url, 15))
            elif (user_emotion) == "fear":
                movie_dict.update(scrape_IMDB(url,15))
            elif (user_emotion) == "angry" :
                movie_dict.update(scrape_IMDB(url,15))
            elif (user_emotion) == "disgust" :
                movie_dict.update(scrape_IMDB(url,10))
            elif (user_emotion) == "surprise" :
                movie_dict.update(scrape_IMDB(url,10))
        elif "www.rottentomatoes.com" in url:
            if (user_emotion) == "happy":
                movie_dict.update(scrape_rt(url, 20))
            elif (user_emotion) == "sad":
                movie_dict.update(scrape_rt(url, 20))
            elif (user_emotion) == "neutral":
                movie_dict.update(scrape_rt(url, 15))
            elif (user_emotion) == "fear" :
                movie_dict.update(scrape_rt(url,15))
            elif (user_emotion) == "angry" :
                movie_dict.update(scrape_rt(url,10))
            elif (user_emotion) == "surprise" :
                movie_dict.update(scrape_rt(url,10))
            elif (user_emotion) == "disgust" :
                movie_dict.update(scrape_rt(url,10))
            
    movie_dict = rank_movies(movie_dict)
    
    # load movie page
    root = Tk()
    movie_page = movie_page.movie_page(root, movie_dict)
    root.mainloop()

    # Cosine-Similarity analysis
    userClicked = movie_page.selected_movie
    userClicked = list(set(userClicked))
    movieName = userClicked[0]

    summary_list = get_movie_summary(movie_dict, movieName)

    targetIndex = find3MostSim(movie_dict, summary_list)
    targetMovies = []
    targetMovieSummary = []
    mainSummary = summary_list[1]
    for i in targetIndex:
        summary = summary_list[0][i]
        targetMovieSummary.append(summary)
        for key, value in movie_dict.items():
            if summary == value[0]:
                targetMovies.append(key)

    #userClicked = movie_page.selected_movie
    



    # load summary page based on users' selection
    SP = Tk()
    Summary_Page = summary_page.Summary_Page(SP, targetMovies, targetMovieSummary, mainSummary)
    SP.mainloop()


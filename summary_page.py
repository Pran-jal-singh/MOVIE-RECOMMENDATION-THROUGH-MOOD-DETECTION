from tkinter import *
from tkinter.ttk import *

class Summary_Page(object):
	def __init__(self, window, targetMovies, targetMovieSummary, mainSummary):
		self.window = window
		self.window.title('Summary Page')
		self.window.geometry("700x500")
		
		self.mainSummary = mainSummary
		self.targetMovies = targetMovies
		self.targetMovieSummary = targetMovieSummary

		#add label
		self.label = Label(self.window, text = "Summary of the movie you selected")
		self.label.config(font =("Courier New", 15,"bold"))  
		self.label.pack()

		# show summary of the movie on tkinter window
		text = Text(self.window, height = 18, width = 70, bg = "orange")
		text.insert(INSERT, "\n\n\n\n\n\n"+self.mainSummary)
		text.pack()

		#add label
		self.label_2 = Label(self.window, text = "Based on your search, here we recommend you to check out following movies:")
		self.label_2.config(font =("Courier New", 15,"bold"))  
		self.label_2.pack()
		recText = Text(self.window, height = 30,  width = 70, bg = "light blue") 
		for movie, summary in zip(self.targetMovies, self.targetMovieSummary):
			recText.insert(INSERT, "\n\n\n🎬🎞" + movie + ": ")
			recText.insert(END, summary)
			recText.pack(expand=1, fill=BOTH)


# window = Tk()
# Summary_Page(window, targetMovies, targetMovieSummary, mainSummary)
# window.mainloop()recText = Text(self.window, height = 30,  width = 70, bg = "light"

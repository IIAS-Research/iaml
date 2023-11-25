

class Citation:
    def __init__(self, title, authors, date, review=None, doi=None, abstract=None):
        self.title = title
        self.authors = authors
        self.date = date
        self.review = review
        self.doi = doi
        self.abstract = abstract
        
    def str(self):
        self.cite()
        
    def cite(self, format="vancouver"):
        if format == "vancouver":
            str_authors = ", ".join(self.authors)
            return f"{str_authors} {self.title} {self.review} {self.date}"
        else:
            return "unknown format"
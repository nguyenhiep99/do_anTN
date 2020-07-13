class Config:
    def __init__(self, C=1.0, degree=3, alpha=1.0, kernel="linear"):
        self.C = C
        self.degree = degree
        self.alpha = alpha
        self.kernel = kernel

        self.__check()

    def __check(self):
        if self.C == 0:
            self.C = 1.0

        if self.degree == 0:
            self.degree = 3

        if self.alpha == 0:
            self.alpha = 1.0

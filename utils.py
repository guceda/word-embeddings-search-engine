import gensim

def tokenize(document):
    return list(gensim.utils.tokenize(document.lower()))


def show_scores(documents, scores, n=10):
    for i in range(n):
        print("======== RANK: {} | SCORE: {} =======".format(i + 1, scores[i]))
        print(documents[i])
        print("")
    print("\n")

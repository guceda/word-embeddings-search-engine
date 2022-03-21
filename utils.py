import gensim

def tokenize(document):
    return list(gensim.utils.tokenize(document.lower()))

def show_scores(documents, scores, n=10):
    for i in range(n):
        print(f"======== RANK: {i + 1} | SCORE: {scores[i]} | TEXT: {documents[i]['metadata']} =======")
        #print(documents[i]['metadata'])
        # print("")
    #print("\n")

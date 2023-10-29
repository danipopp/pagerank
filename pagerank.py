import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    #if len(sys.argv) != 2:
        #sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl("corpus2")#crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    vis = {}
    for name in corpus:
        vis[name]=0

    # if no links
    if len(corpus[page])==0:
        for name in vis:
            vis[name]=1/len(corpus)
        return vis
    
    # picking random page
    randoms = (1-damping_factor)/len(corpus)

    # picking link
    links = damping_factor/len(corpus[page])

    # add prob
    for name in vis:
        vis[name] += randoms
        
        if name in corpus[page]:
            vis[name] += links

    return vis


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # init
    vis = {}
    for name in corpus:
        vis[name]=0
    # choose first page randomly
    page = random.choice(list(vis))
    vis[page]+=1

    for i in range(0,n-1):
        # write into the transition_model
        transModel = transition_model(corpus,page,damping_factor)

        # pick a random page
        ran=random.random()
        prob=0

        for name,proba in transModel.items():
            prob+=proba
            if ran<=prob:
                page=name
                break

        vis[page]+=1

    ranks = {}
    for name, numVis in vis.items():
        ranks[name]=numVis/n

    print('Sum of sample page ranks: ', round(sum(ranks.values()), 4))

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    ranks = {}
    for name in corpus:
        ranks[name]=1/len(corpus)
    nranks = {}
    for name in corpus:
        nranks[name]=None

    runs = 0
    mrc = 1/len(corpus)
    while mrc > 0.001:
        runs+=1
        mrc = 0
    
        for name in corpus:
            prob = 0
            for nextName in corpus:
                if len(corpus[nextName])==0:
                    prob+=ranks[nextName]*(1/len(corpus))
                elif name in corpus[nextName]:
                    prob+=ranks[nextName]/len(corpus[nextName])
            nRank = ((1 - damping_factor) / len(corpus)) + (damping_factor * prob)
            nranks[name]=nRank
        
        nFaktor = sum(nranks.values())
        for name,rank in nranks.items():
            nranks[name]=rank/nFaktor
        
        for name in corpus:
            rchange=abs(ranks[name]-nranks[name])
            if rchange > mrc:
                mrc = rchange
        
        ranks = nranks.copy()

    return ranks


if __name__ == "__main__":
    main()

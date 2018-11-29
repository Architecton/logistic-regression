import pylab
import numpy as np
import solution


def draw_decision(X, y, classifier, at1, at2, lam, grid=50):

    points = np.take(X, [at1, at2], axis=1)
    maxx, maxy = np.max(points, axis=0)
    minx, miny = np.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02*difx
    minx -= 0.02*difx
    maxy += 0.02*dify
    miny -= 0.02*dify

    for c,(x,y) in zip(y,points):
        pylab.text(x,y,str(c), ha="center", va="center")
        pylab.scatter([x],[y],c=["b","r"][c!=0], s=200)

    num = grid
    prob = np.zeros([num, num])
    for xi,x in enumerate(np.linspace(minx, maxx, num=num)):
        for yi,y in enumerate(np.linspace(miny, maxy, num=num)):
            # probability of the closest example
            diff = points - np.array([x,y])
            dists = (diff[:,0]**2 + diff[:,1]**2)**0.5 # euclidean
            ind = np.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pylab.imshow(prob, extent=(minx,maxx,maxy,miny), cmap="seismic")

    pylab.xlim(minx, maxx)
    pylab.ylim(miny, maxy)
    pylab.xlabel(at1)
    pylab.ylabel(at2)

    pylab.title('Lambda = {0}'.format(lam), fontsize=20)
    pylab.show()


X,y = solution.load('reg.data')

lam = 2.3
learner = solution.LogRegLearner(lambda_=lam)
classifier = learner(X,y)

draw_decision(X, y, classifier, 0, 1, lam)
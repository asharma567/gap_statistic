import scipy
from  scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans as k_means
import joblib
 
dst = euclidean

k_means_args_dict = {
    'n_clusters': 0,
    # drastically saves convergence time
    'init':'k-means++',
    'max_iter':100,
    'n_init':1,
    'verbose':False,
    # 'n_jobs':8
        }

def gap(data, refs=None, nrefs=20, ks=range(1,11)):
    """
    I: NumPy array, reference matrix, number of reference boxes, number of clusters to test
    O: Gaps NumPy array, Ks input list
    
    Give the list of k-values for which you want to compute the statistic in ks. By Gap Statistic 
    from Tibshirani, Walther.
    """
    shape = data.shape
    
    if not refs: 
        tops = data.max(axis=0)
        bottoms = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops - bottoms))
        rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i] * dists + bottoms
    else:
        rands = refs
 
    gaps = scipy.zeros((len(ks),))
    
    for (i,k) in enumerate(ks):
        k_means_args_dict['n_clusters'] = k
        kmeans = k_means(**k_means_args_dict)
        kmeans.fit(data)
        (cluster_centers, point_labels) = kmeans.cluster_centers_, kmeans.labels_

        disp = sum([dst(data[current_row_index, :], cluster_centers[point_labels[current_row_index],:]) for current_row_index in range(shape[0])])

        refdisps = scipy.zeros((rands.shape[2],))

        for j in range(rands.shape[2]):

            kmeans = k_means(**k_means_args_dict)
            kmeans.fit(rands[:, : ,j])
            (cluster_centers, point_labels) = kmeans.cluster_centers_, kmeans.labels_
            refdisps[j] = sum([dst(rands[current_row_index,:,j], cluster_centers[point_labels[current_row_index],:]) for current_row_index in range(shape[0])])
        
        #let k be the index of the array 'gaps'
        gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)
    
    return ks, gaps

if __name__ == '__main__':
    
    #Example of usage on a tfidf matrix
    feature_M = joblib.load('../feature_M.pkl')
    args, gap_scores = gap(feature_M, ks=range(3,10) 
    
    print args, gap_scores
    joblib.dump(args, '../args.joblib')
    joblib.dump(gap_scores, '../gap_scores.joblib')
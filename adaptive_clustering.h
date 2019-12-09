#include <string>
#include <armadillo>

/**
 * @file adaptive_clustering.h
 */

#ifndef ADAPTIVECLUSTERING_ADAPTIVE_CLUSTERING_H
#define ADAPTIVECLUSTERING_ADAPTIVE_CLUSTERING_H

using namespace std;
using namespace arma;

/**
 * @struct cluster_report
 */
typedef struct cluster_report {
    mat centroids;      /**< Mat (structure from Armadillo library). */
    int k;              /**< The number of clusters. */
    double BetaCV;      /**< A metric (double value) used to choose the optimal number of clusters through the Elbow criterion. */
    int *cidx;          /**< A matrix [1, n_data] indicating the membership of data to a cluster through an index. @see create_cidx_matrix for matrix generation */
} cluster_report;

/**
 * The algorithm calculates the number of rows and columns in a CSV file and saves the information into dim and data.
 * @param [in] fname the path referred to a CSV file.
 * @param [in,out] dim the number of columns in the CSV file.
 * @param [in,out] data the number of rows in the CSV file.
 * @return 0 if file is read correctly, otherwise -9;
 */
extern int getDatasetDims(string fname, int &dim, int &data);
/**
 * The algorithm loads a CSV file into a matrix of double.
 * @param [in] fname the number of columns in the CSV file.
 * @param [in,out] array a matrix [n_data, n_dims], in which the CSV file is loaded.
 * @param [in] n_dims the number of columns in the CSV file.
 * @return 0 if the dataset is loaded correctly, -2 if array is NULL, -8 if there is a
 *          conversion error on a read value, -9 if there is an error with inputFile.
 */
extern int loadData(string fname, double **array, int n_dims);
/**
 * Get the mean value from an array of n elements.
 * This function exits with code -2 if arr is NULL.
 * @param [in] arr the array containing the data.
 * @param [in] n_data the number of elements in the array.
 * @return a double value indicating the mean.
 */
extern double getMean(double *arr, int n_data);
/**
 * This function centers each dimension in the dataset around the mean value.
 * @param [in, out] data a matrix [n_dims, n_data] containing the dataset.
 * @param [in] n_dims the number of dimensions.
 * @param [in] n_data the number of data.
 * @return 0 if it is correct, otherwise -2 if data in NULL
 */
extern int Standardize_dataset(double **data, int n_dims, int n_data);
/**
 * This function computes the Pearson coefficient of given dimensions centered around the mean value.
 * This function exits with code -2 if X or Y are NULL.
 * @param [in] X an array containing the data of the first dimension.
 * @param [in] Y an array containing the data of the first dimension.
 * @param [in] n_data the number of data in each array.
 * @return a double value indicating the Pearson coefficient between X and Y.
 */
extern double PearsonCoefficient(double *X, double *Y, int n_data);
/**
  * This function computes the Pearson matrix.
  * @param [in, out] pearson a matrix [n_dims, n_dims] to store Pearson coefficients.
  * @param [in] data a matrix [n_dims, n_data] on which computes the Pearson coefficients.
  * @param [in] n_data the number of data.
  * @param [in] n_dims the number of dimensions.
  * @return 0 if it is correct, -2 if pearson or data are NULL
  */
extern int computePearsonMatrix(double **pearson, double **data, int n_data, int n_dims);
/**
 * This function says if a dimension have to enter in CORR set computing
 * the overall Pearson coefficient (row-wise sum of the Pearson coefficients)
 * for a given dimension. The function exit with code -2 if pcc is NULL.
 * @param [in] ndims the number of dimensions.
 * @param [in] dimensionID an index between 0 and ndims indicating the given dimension.
 * @param [in] pcc the matrix [ndims, ndims] containing the Pearson coefficients.
 * @return true if the dimension have to enter in CORR, else false.
 */
extern bool isCorrDimension(int ndims, int dimensionID, double **pcc);
/**
 * This function computes the number of dimensions have to enter in CORR
 * and UNCORR sets.
 * @param [in] pcc the matrix [ndims, ndims] containing the Pearson coefficients.
 * @param [in] ndims the number of dimensions.
 * @param [in,out] corr_vars an integer value which will contain the cardinality
 *                           of CORR.
 * @param [in,out] uncorr_vars an integer value which will contain the
 *                              cardinality of UNCORR.
 * @return 0 if it is correct, -2 if pcc is NULL.
 */
extern int computeCorrUncorrCardinality(double **pcc, int ndims, int &corr_vars, int &uncorr_vars);
/**
 * This function computes the Principal Component Analysis with the 2 Principal Components.
 * @param [in] data_to_transform a matrix [data_dim, n_data] containing the data to transform.
 * @param [in] data_dim the number of dimensions in data_to_transform.
 * @param [in] n_data the number of data in data_to_transform.
 * @param [in, out] new_space the subspace resulting from the input data transformed through the 2 Principal Components.
 * @return 0 if it is correct, -2 if data_to_transform or new_space are NULL, -1 for memory allocation
 *          error on covariance matrix.
 */
extern int PCA_transform(double **data_to_transform, int data_dim, int n_data, double **new_space);
/**
 * The algorithm calculates the number of elements of an indicated cluster in the given cluster report.
 * This function exits with code -2 if rep.cidx is NULL.
 * @param [in] rep a cluster_report structure describing the K-Means instance carried out.
 * @param [in] cluster_id a number (between 0 and rep.k) indicating the cluster whereby we want to know the number of data in it.
 * @param [in] n_data number of rows in the dataset matrix.
 * @return an integer indicating the number of data in the cluster with the given id.
 */
extern int cluster_size(cluster_report rep, int cluster_id, int n_data);
/**
 * This function runs the K-Means with the elbow criterion; it tries different number of clusters and
 * evaluates the instance of K-Means through the BetaCV metric in order to find the optimal number of clusters.
 * This function exits with code -2 if dataset is NULL.
 * @param [in] dataset a matrix [2, n_data] containing the data for K-Means run.
 * @param [in] n_data the number of data in dataset.
 * @param [in] k_max the max number of clusters to try for elbow criterion.
 * @param [in] elbow_thr the error tolerance for BetaCV metric.
 * @return a cluster_report containing the result of the K-Means run.
 */
extern cluster_report run_K_means(double **dataset, int n_data, long k_max, double elbow_thr);
/**
 * The algorithm creates the array [1, N_DATA] which indicates the membership of each data to a cluster through an integer.
 * @param [in] data a matrix [n_data, n_dims] on which the K-Means run was made.
 * @param [in] n_data number of rows in the dataset matrix.
 * @param [in,out] instance a cluster_report structure describing the K-Means instance carried out.
 * @return 0 if it is correct, otherwise-2 if data or instance.cidx are NULL or instance.centroids is empty.
 */
extern int create_cidx_matrix(double **data, int n_data, cluster_report instance);
/**
 * This function computes the intra-cluster weight, the sum of the Euclidean distance
 * between the centroids of a cluster and each data in it.
 * This function exits with code -2 if data or instance.cidx are NULL or instance.centroids is empty.
 * @param [in] data the data on which the K-Means was executed.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data in data.
 * @return a double value representing the intra-cluster weight.
 */
extern double WithinClusterWeight(double **data, cluster_report instance, int n_data);
/**
 * This function computes the inter-cluster weight, the weighted sum of the Euclidean distance
 * between the centroids of each cluster and the mean value of the dataset. The weight is the number
 * of points in each cluster.
 * This function exits with code -2 if data is NULL or instance.centroids is empty.
 * @param [in] data the data on which the K-Means was executed.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data in data.
 * @return a double value representing the inter-cluster weight.
 */
extern double BetweenClusterWeight(double **data, cluster_report instance, int n_data);
/**
 * This function computes the number of intra-cluster pairs.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data on which the K-Means was executed.
 * @return an integer indicating the number of intra-cluster pairs.
 */
extern int WithinClusterPairs(cluster_report instance, int n_data);
/**
 * This function computes the number of inter-cluster pairs.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data on which the K-Means was executed.
 * @return an integer indicating the number of inter-cluster pairs.
 */
extern int BetweenClusterPairs(cluster_report instance, int n_data);
/**
 * This function computes the BetaCV metric for elbow criterion evaluation.
 * This function exits with code -2 if data is NULL.
 * @param [in] data a matrix containing the data on which the K-Means run was executed.
 * @param [in] instance the cluster_report containing the information about the K-Means run.
 * @param [in] n_data the number of data.
 * @return a double value indicating the BetaCV value.
 */
extern double BetaCV(double **data, cluster_report instance, int n_data);
/**
 * The algorithm calculates the Euclidean distance between two points p_c and p_1.
 * @param [in] xc the first component of p_c.
 * @param [in] yc the second component of p_c.
 * @param [in] x1 the first component of p_1.
 * @param [in] y1 the second component of p_1.
 * @return a double value indicating the Euclidean distance between the two given points.
 */
extern double L2distance(double xc, double yc, double x1, double y1);
/**
 * This function computes the distance between each point (not discarded previously)
 * and its centroids and it is stored in "dist". Moreover the cardinality of the cluster
 * is computed. This function exits with code -2 if incircle, subspace or rep.cidx are NULL
 * or rep.centroids is empty.
 * @param [in] rep a cluster_report containing the KMeans run information.
 * @param [in] n_data the number of data.
 * @param [in] clusterID the index of the cluster.
 * @param [in] incircle an array [1, n_data] indicating the discarded points.
 * @param [in] subspace a matrix [2, n_data] on which KMeans was executed.
 * @param [in,out] dist the distance between points and centroids.
 * @return the cardinality of the given cluster (clusterID).
 */
extern int computeActualClusterInfo(cluster_report rep, int n_data, int clusterID, bool *incircle, double **subspace, double &dist);
/**
 * This functions computes the new inliers through the radius and it updates "incircle".
 * This function exits with code -2 if incircle, subspace or rep.cidx are NULL
 * or rep.centroids is empty.
 * @param [in] rep a cluster_report containing the KMeans run information.
 * @param [in] n_data the number of data.
 * @param [in] clusterID the index of the cluster.
 * @param [in] incircle an array [1, n_data] indicating the discarded points.
 * @param [in] subspace a matrix [2, n_data] on which KMeans was executed.
 * @param radius the radius to use for inliers search.
 * @return the elements (not discarded in the previous iterations) considered as inliers.
 */
extern int computeInliers(cluster_report rep, int n_data, int clusterID, bool *incircle, double **subspace, double radius);
/**
 * This function computes the number of times a given point is evaluated as outlier
 * in all the candidate subspaces. This function exits with code -2 if incircle is NULL.
 * @param [in] incircle an array [1, n_data] indicating the discarded points.
 * @param [in] uncorr_vars the number of candidate subspaces.
 * @param [in] data_idx the index of point to be evaluated.
 * @return the number of times a given point is evaluated as outlier.
 */
extern int countOutliers(bool **incircle, int uncorr_vars, int data_idx);

#endif //ADAPTIVECLUSTERING_ADAPTIVE_CLUSTERING_H
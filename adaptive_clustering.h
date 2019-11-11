#include <string>
#include <armadillo>

/**
 * @file adaptive_clustering.h
 */

#ifndef ADAPTIVECLUSTERING_ADAPTIVE_CLUSTERING_H
#define ADAPTIVECLUSTERING_ADAPTIVE_CLUSTERING_H

#endif //ADAPTIVECLUSTERING_ADAPTIVE_CLUSTERING_H

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

extern void getDatasetDims(string fname, int *dim, int *data);
extern int loadData(string fname, double **array, int n_dims);
extern double getMean(double *arr, int n_data);
extern void Standardize_dataset(double **data, int n_dims, int n_data);
extern double PearsonCoefficient(double *X, double *Y, int n_data);
extern void PCA_transform(double **data_to_transform, int data_dim, int n_data, double **new_space);
extern int cluster_size(cluster_report rep, int cluster_id, int n_data);
extern cluster_report run_K_means(double **dataset, int n_data, long k_max, double elbow_thr);
extern void create_cidx_matrix(double **data, int n_data, cluster_report instance);
extern double WithinClusterWeight(double **data, cluster_report instance, int n_data);
extern double BetweenClusterWeight(double **data, cluster_report instance, int n_data);
extern int WithinClusterPairs(cluster_report instance, int n_data);
extern int BetweenClusterPairs(cluster_report instance, int n_data);
extern double BetaCV(double **data, cluster_report instance, int n_data);
extern double L2distance(double xc, double yc, double x1, double y1);
extern void csv_out_info(double **data, int n_data, string name, string outdir, bool *incircle, cluster_report report);
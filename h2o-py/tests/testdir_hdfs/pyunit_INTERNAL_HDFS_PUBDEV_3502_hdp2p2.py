from __future__ import print_function
import sys
sys.path.insert(1,"../../")
import h2o
import time
from tests import pyunit_utils
from h2o.transforms.decomposition import H2OPCA
import numpy
#----------------------------------------------------------------------
# This test is carried out to find out if remove the new memory
# allocations will work faster.
#----------------------------------------------------------------------


def hdfs_import_bigCat():

    # Check if we are running inside the H2O network by seeing if we can touch
    # the namenode.
    hadoop_namenode_is_accessible = pyunit_utils.hadoop_namenode_is_accessible()

    if hadoop_namenode_is_accessible:
        numTimes = 10
        hdfs_name_node = pyunit_utils.hadoop_namenode()
        hdfs_cross_file = "/datasets/la1s.wc.arff.txt.zip"
        url = "hdfs://{0}{1}".format(hdfs_name_node, hdfs_cross_file)
        cross_h2o = h2o.import_file(url)
        cross_h2o.drop("CLASS_LABEL")
        x= cross_h2o.names
        runtimes = []

        for ind in range(numTimes):
            randomizedPCA = H2OPCA(k=1938, transform="STANDARDIZE", pca_method="Randomized", compute_metrics=True,
                                    use_all_factor_levels=True, max_iterations=10)
            randomizedPCA.train(x=x, training_frame=cross_h2o)
            runtimes.append(randomizedPCA._model_json["output"]["run_time"])
            print("Run time in (ms) is {0}".format(runtimes))
            h2o.remove(randomizedPCA)       # remove model to save space just in case

        # write out summary run results
        print("*******************************")
        print("All run times {0}".format(runtimes))
        arr = numpy.array(runtimes)
        print(" Maximum run time is {0}, minimum run time is {1}".format(max(runtimes), min(runtimes)))
        print("Mean run time is {0}, std is {1}".format(numpy.mean(arr, axis=0), numpy.std(arr, axis=0)))
        print("*******************************")
    else:
        raise EnvironmentError



if __name__ == "__main__":
    pyunit_utils.standalone_test(hdfs_import_bigCat)
else:
    hdfs_import_bigCat()

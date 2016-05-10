#include <Mfpfh/long_headers.h>


class cbFPFH
{

public :

    pcl::PointCloud<pcl::PointXYZ> cloud1, cloud2;
    pcl::PointCloud<pcl::Normal> cloud1_normals, cloud2_normals;
    pcl::PointCloud<pcl::PointXYZ> cloud1_keypoints, cloud2_keypoints;

    pcl::PointCloud<pcl::FPFHSignature33> cloud1_fpfh, cloud2_fpfh;

    std::vector<int> cloud1_keypoints_indices, cloud2_keypoints_indices;

    pcl::IndicesConstPtr indices;

    pcl::PolygonMesh mesh1, mesh2;

    void calculate_normals ( float radius )
    {
        // Estimate the normals.
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setRadiusSearch(radius);
        normalEstimation.setNumberOfThreads(12);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod(kdtree);

        normalEstimation.setInputCloud(cloud1.makeShared());
        normalEstimation.compute(cloud1_normals);

        normalEstimation.setInputCloud(cloud2.makeShared());
        normalEstimation.compute(cloud2_normals);
    }

    void  calculate_iss_keypoints_for_3DLRF ( float model_resolution)
    {

        //
        //  ISS3D parameters
        //
        double iss_salient_radius_;
        double iss_non_max_radius_;
        double iss_normal_radius_;
        double iss_border_radius_;
        double iss_gamma_21_ (0.8);
        double iss_gamma_32_ (0.8);
        double iss_min_neighbors_ (5);
        int iss_threads_ (8);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());

        // Fill in the model cloud

        // Compute model_resolution

        iss_salient_radius_ = 10 * model_resolution;
        iss_non_max_radius_ = 6 * model_resolution;
        iss_normal_radius_ = 6 * model_resolution;
        iss_border_radius_ = 2 * model_resolution;

        //
        // Compute keypoints
        //
        pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

        iss_detector.setSearchMethod (tree);
        iss_detector.setSalientRadius (iss_salient_radius_);
        iss_detector.setNonMaxRadius (iss_non_max_radius_);

        iss_detector.setNormalRadius (iss_normal_radius_);
        iss_detector.setBorderRadius (iss_border_radius_);

        iss_detector.setThreshold21 (iss_gamma_21_);
        iss_detector.setThreshold32 (iss_gamma_32_);
        iss_detector.setMinNeighbors (iss_min_neighbors_);
        iss_detector.setNumberOfThreads (iss_threads_);


        iss_detector.setInputCloud (cloud1.makeShared());
        iss_detector.compute (cloud1_keypoints);

        iss_detector.setInputCloud (cloud2.makeShared());
        iss_detector.compute (cloud2_keypoints);

    }



    void get_keypoint_indices()
    {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud (cloud1.makeShared());
        pcl::PointXYZ searchPoint;

        for (int i = 0; i < cloud1_keypoints.size(); i++)
        {
            searchPoint = cloud1_keypoints[i];

            int K = 1;

            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);

            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
            {
                cloud1_keypoints_indices.push_back(pointIdxNKNSearch[0]);
            }

        }

        kdtree.setInputCloud (cloud2.makeShared());

        for (int i = 0; i < cloud2_keypoints.size(); i++)
        {
            searchPoint = cloud2_keypoints[i];

            int K = 1;

            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);

            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
            {
                cloud2_keypoints_indices.push_back(pointIdxNKNSearch[0]);
            }

        }

    }



    void calculate_12_dim_FPFH ( float radius )
    {

        // 12 DIMENSIONAL FPFH

        // SHOT estimation object.
        pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setRadiusSearch(radius);

        fpfh.setNrSubdivisions(4,4,4);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        fpfh.setSearchMethod(kdtree);

        fpfh.setInputCloud(cloud1_keypoints.makeShared());
        fpfh.setSearchSurface(cloud1.makeShared());
        fpfh.setInputNormals(cloud1_normals.makeShared());
        fpfh.compute(cloud1_fpfh);


        fpfh.setInputCloud(cloud2_keypoints.makeShared());
        fpfh.setSearchSurface(cloud2.makeShared());
        fpfh.setInputNormals(cloud2_normals.makeShared());
        fpfh.compute(cloud2_fpfh);



        /* FULL 33 DIMENSIONAL FPFH
 * /

    /*
     *  pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setRadiusSearch(radius);

        fpfh.setNumberOfThreads(12);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        fpfh.setSearchMethod(kdtree);

        fpfh.setInputCloud(cloud1_keypoints.makeShared());
        fpfh.setSearchSurface(cloud1.makeShared());
        fpfh.setInputNormals(cloud1_normals.makeShared());
        fpfh.setNumberOfThreads(12);
        fpfh.compute(cloud1_fpfh);


        fpfh.setInputCloud(cloud2_keypoints.makeShared());
        fpfh.setSearchSurface(cloud2.makeShared());
        fpfh.setInputNormals(cloud2_normals.makeShared());
        fpfh.setNumberOfThreads(12);
        fpfh.compute(cloud2_fpfh);

        */
    }


};






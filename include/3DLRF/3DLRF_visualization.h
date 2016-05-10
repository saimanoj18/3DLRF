#include <3DLRF/3DLRF_headers.h>


class patch_descriptor
{
public:
    std::vector<float> vector;
};


class threeDLRF
{
public:

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZ> cloud_keypoints;

    std::vector<patch_descriptor> cloud_patch_descriptors;
    std::vector<patch_descriptor> cloud_LRF_descriptors;
    std::vector<patch_descriptor> cloud_REFERENCE_FRAME_descriptors;

    pcl::PointCloud<pcl::PointXYZ> encoded3dkeypoints;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;


    std::vector<int> cloud_keypoints_indices;
    std::vector<int> patch_descriptor_indices;

    void voxelize_cloud_005(float radius)
    {
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setLeafSize(radius, radius, radius);
        voxel_grid.setInputCloud(cloud.makeShared());
        voxel_grid.filter(cloud);
    }

    void detect_uniform_keypoints_on_cloud(float keypoint_radius)
    {
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setLeafSize(keypoint_radius, keypoint_radius, keypoint_radius);
        voxel_grid.setInputCloud(cloud.makeShared());
        voxel_grid.filter(cloud_keypoints);
    }


    void  calculate_iss_keypoints ( float model_resolution)
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


        iss_detector.setInputCloud (cloud.makeShared());
        iss_detector.compute (cloud_keypoints);



    }




    void JUST_REFERENCE_FRAME_descriptors(float patch_radius)
    {
        for (int i = 0; i < cloud_keypoints.size(); i++)
        {
            pcl::PointXYZ currentPoint = cloud_keypoints[i];

            std::vector<int> nn_indices;
            std::vector<float> nn_sqr_distances;

            if(kdtree.radiusSearch(currentPoint,patch_radius,nn_indices,nn_sqr_distances) > 0)
            {
                pcl::PointCloud<pcl::PointXYZ> raw_patch;
                pcl::copyPointCloud(cloud, nn_indices, raw_patch);

                /// Removal of Minimum point...offset
                Eigen::Vector4f min_raw_patch, max_raw_patch;
                pcl::getMinMax3D( raw_patch, min_raw_patch, max_raw_patch);

                for (int j = 0; j < raw_patch.size(); j++)
                {
                    raw_patch.points[j].x = raw_patch.points[j].x - min_raw_patch[0];
                    raw_patch.points[j].y = raw_patch.points[j].y - min_raw_patch[1];
                    raw_patch.points[j].z = raw_patch.points[j].z - min_raw_patch[2];
                }
                //Important to remove the min_raw_patch from currentPoint
                currentPoint.x = currentPoint.x - min_raw_patch[0];
                currentPoint.y = currentPoint.y - min_raw_patch[1];
                currentPoint.z = currentPoint.z - min_raw_patch[2];

                Eigen::Matrix3f LRF;
                get_local_rf(currentPoint, patch_radius, raw_patch, LRF);

                patch_descriptor REFERENCE_FRAME; REFERENCE_FRAME.vector.resize(9);
                for (int ref = 0; ref < 9; ref++)
                    REFERENCE_FRAME.vector[ref] = LRF(ref);

                cloud_REFERENCE_FRAME_descriptors.push_back(REFERENCE_FRAME);

                Eigen::Vector3f point_here =  LRF * currentPoint.getVector3fMap();

                patch_descriptor pt_LRF;

                //cout << transformedPoint.x << endl;
                pt_LRF.vector.push_back(point_here[0]);
                pt_LRF.vector.push_back(point_here[1]);
                pt_LRF.vector.push_back(point_here[2]);

                /// added for visualization
                pcl::PointXYZ encoded; encoded.getVector3fMap() = point_here;
                encoded3dkeypoints.push_back(encoded);

                cloud_LRF_descriptors.push_back(pt_LRF);

                patch_descriptor_indices.push_back(i);
            }

        }

    }





    void get_local_rf (pcl::PointXYZ current_point, float lrf_radius, pcl::PointCloud<pcl::PointXYZ>& cloud_here, Eigen::Matrix3f &rf)
    {

        int current_point_idx;

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud (cloud_here.makeShared());

        std::vector<int> n_indices;
        std::vector<float> n_sqr_distances;

        if ( kdtree.radiusSearch (current_point, lrf_radius, n_indices, n_sqr_distances) > 0 )
        {
            current_point_idx = n_indices[0];

        }

        const Eigen::Vector4f& central_point = (cloud_here)[current_point_idx].getVector4fMap ();

        pcl::PointXYZ searchPoint;
        searchPoint = cloud_here[current_point_idx];

        Eigen::Matrix<double, Eigen::Dynamic, 4> vij (n_indices.size (), 4);

        Eigen::Matrix3d cov_m = Eigen::Matrix3d::Zero ();

        double distance = 0.0;
        double sum = 0.0;

        int valid_nn_points = 0;

        for (size_t i_idx = 0; i_idx < n_indices.size (); ++i_idx)
        {
            Eigen::Vector4f pt = cloud_here.points[n_indices[i_idx]].getVector4fMap ();
            if (pt.head<3> () == central_point.head<3> ())
                continue;

            // Difference between current point and origin
            vij.row (valid_nn_points).matrix () = (pt - central_point).cast<double> ();
            vij (valid_nn_points, 3) = 0;

            distance = lrf_radius - sqrt (n_sqr_distances[i_idx]);

            // Multiply vij * vij'
            cov_m += distance * (vij.row (valid_nn_points).head<3> ().transpose () * vij.row (valid_nn_points).head<3> ());

            sum += distance;
            valid_nn_points++;
        }

        if (valid_nn_points < 5)
        {
            //PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Neighborhood has less than 5 vertexes. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
            rf.setConstant (std::numeric_limits<float>::quiet_NaN ());

            cout <<"\n\n\n\ Something CRAZY is Happening dude!!! \n\n\n"<< endl;
        }

        cov_m /= sum;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver (cov_m);

        const double& e1c = solver.eigenvalues ()[0];
        const double& e2c = solver.eigenvalues ()[1];
        const double& e3c = solver.eigenvalues ()[2];

        if (!pcl_isfinite (e1c) || !pcl_isfinite (e2c) || !pcl_isfinite (e3c))
        {
            //PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Eigenvectors are NaN. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
            rf.setConstant (std::numeric_limits<float>::quiet_NaN ());

            cout <<"\n\n\n\ Something CRAZY is Happening dude!!! \n\n\n"<< endl;
        }

        // Disambiguation
        Eigen::Vector4d v1 = Eigen::Vector4d::Zero ();
        Eigen::Vector4d v3 = Eigen::Vector4d::Zero ();
        v1.head<3> ().matrix () = solver.eigenvectors ().col (2);
        v3.head<3> ().matrix () = solver.eigenvectors ().col (0);

        int plusNormal = 0, plusTangentDirection1=0;
        for (int ne = 0; ne < valid_nn_points; ne++)
        {
            double dp = vij.row (ne).dot (v1);
            if (dp >= 0)
                plusTangentDirection1++;

            dp = vij.row (ne).dot (v3);
            if (dp >= 0)
                plusNormal++;
        }

        //TANGENT
        plusTangentDirection1 = 2*plusTangentDirection1 - valid_nn_points;
        if (plusTangentDirection1 == 0)
        {
            int points = 5; //std::min(valid_nn_points*2/2+1, 11);
            int medianIndex = valid_nn_points/2;

            for (int i = -points/2; i <= points/2; i++)
                if ( vij.row (medianIndex - i).dot (v1) > 0)
                    plusTangentDirection1 ++;

            if (plusTangentDirection1 < points/2+1)
                v1 *= - 1;
        }
        else if (plusTangentDirection1 < 0)
            v1 *= - 1;

        //Normal
        plusNormal = 2*plusNormal - valid_nn_points;
        if (plusNormal == 0)
        {
            int points = 5; //std::min(valid_nn_points*2/2+1, 11);
            int medianIndex = valid_nn_points/2;

            for (int i = -points/2; i <= points/2; i++)
                if ( vij.row (medianIndex - i).dot (v3) > 0)
                    plusNormal ++;

            if (plusNormal < points/2+1)
                v3 *= - 1;
        } else if (plusNormal < 0)
            v3 *= - 1;

        rf.row (0).matrix () = v1.head<3> ().cast<float> ();
        rf.row (2).matrix () = v3.head<3> ().cast<float> ();
        rf.row (1).matrix () = rf.row (2).cross (rf.row (0));

    }




    void transformPatchCloud (pcl::PointCloud<pcl::PointXYZ>& input_patch_cloud, /*const pcl::PointXYZ& center_point,*/ const Eigen::Matrix3f& matrix, pcl::PointCloud<pcl::PointXYZ>& transformed_cloud) const
    {
        const unsigned int number_of_points = static_cast <unsigned int> (input_patch_cloud.size ());
        transformed_cloud.points.resize (number_of_points, pcl::PointXYZ ());

        for (unsigned int index_t = 0; index_t < number_of_points; index_t++)
        {
            Eigen::Vector3f transformed_point (
                        input_patch_cloud.points[index_t].x /*- center_point.x*/,
                        input_patch_cloud.points[index_t].y /*- center_point.y*/,
                        input_patch_cloud.points[index_t].z /*- center_point.z*/);

            transformed_point = matrix * transformed_point;

            pcl::PointXYZ new_point;
            new_point.x = transformed_point (0);
            new_point.y = transformed_point (1);
            new_point.z = transformed_point (2);
            transformed_cloud.points[index_t] = new_point;
        }
    }



    // end of class

};





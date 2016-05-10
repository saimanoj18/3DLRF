#include<3DLRF/3DLRF_visualization.h>

int main()
{

    pcl::PointCloud<pcl::PointXYZ> cloud2, cloud1;

    pcl::io::loadPCDFile<pcl::PointXYZ>("../datasets/scene005_0.pcd", cloud2);

    //pcl::io::loadPCDFile<pcl::PointXYZ>("../datasets/Doll018_0.pcd", cloud1);
    pcl::io::loadPCDFile<pcl::PointXYZ>("../datasets/mario000_0.pcd", cloud1);

    threeDLRF RP1, RP2;

    RP1.cloud = cloud1;
    RP1.detect_uniform_keypoints_on_cloud(0.01);
    cout << "Keypoints : " << RP1.cloud_keypoints.size() << endl;


    RP2.cloud = cloud2;
    RP2.detect_uniform_keypoints_on_cloud(0.01);
    cout << "Keypoints : " << RP2.cloud_keypoints.size() << endl;



    clock_t start1, end1;
    double cpu_time_used1;
    start1 = clock();

    // setup
    RP1.kdtree.setInputCloud(RP1.cloud.makeShared());
    RP2.kdtree.setInputCloud(RP2.cloud.makeShared());



    RP1.JUST_REFERENCE_FRAME_descriptors(0.06);
    RP2.JUST_REFERENCE_FRAME_descriptors(0.06);

    end1 = clock();
    cpu_time_used1 = ((double) (end1 - start1)) / CLOCKS_PER_SEC;
    cout <<  "Time taken for Feature Descriptor Extraction: " << (double)cpu_time_used1 << "\n";


    pcl::Correspondences corrs;



    //////////////////////////////////////////////////////////////////////////////////////
    /// \brief kdtree_LRF----------JUST REFERENCE FRAME descriptors
    //////////////////////////////////////////////////////////////////////////////////////
    clock_t start_shot2, end_shot2;
    double cpu_time_used_shot2;
    start_shot2 = clock();

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_LRF;
    pcl::PointCloud<pcl::PointXYZ> pcd_LRF;
    for (int i = 0; i < RP2.cloud_LRF_descriptors.size(); i++)
    {
        pcl::PointXYZ point;
        point.x = RP2.cloud_LRF_descriptors[i].vector[0];
        point.y = RP2.cloud_LRF_descriptors[i].vector[1];
        point.z = RP2.cloud_LRF_descriptors[i].vector[2];

        pcd_LRF.push_back(point);
    }

    kdtree_LRF.setInputCloud(pcd_LRF.makeShared());
    for (int i = 0; i < RP1.cloud_LRF_descriptors.size(); i++)
    {
        pcl::PointXYZ searchPoint;
        searchPoint.x = RP1.cloud_LRF_descriptors[i].vector[0];
        searchPoint.y = RP1.cloud_LRF_descriptors[i].vector[1];
        searchPoint.z = RP1.cloud_LRF_descriptors[i].vector[2];

        std::vector<int> nn_indices;
        std::vector<float> nn_sqr_distances;

        if (kdtree_LRF.radiusSearch(searchPoint,0.02,nn_indices,nn_sqr_distances) > 0)// IMPORTANT PARAMETER 0.2m or 0.1m ...
        {
            std::vector<double> angles_vector;
            for (int j = 0; j < nn_indices.size(); j++)
            {

                Eigen::Vector4f rf1_x,rf1_y,rf1_z,rf2_x,rf2_y,rf2_z;
                rf1_x[0] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[0];
                rf1_x[1] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[1];
                rf1_x[2] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[2];
                rf1_x[3] = 0;

                rf1_y[0] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[3];
                rf1_y[1] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[4];
                rf1_y[2] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[5];
                rf1_y[3] = 0;

                rf1_z[0] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[6];
                rf1_z[1] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[7];
                rf1_z[2] = RP1.cloud_REFERENCE_FRAME_descriptors[i].vector[8];
                rf1_z[3] = 0;

                rf2_x[0] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[0];
                rf2_x[1] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[1];
                rf2_x[2] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[2];
                rf2_x[3] = 0;

                rf2_y[0] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[3];
                rf2_y[1] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[4];
                rf2_y[2] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[5];
                rf2_y[3] = 0;

                rf2_z[0] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[6];
                rf2_z[1] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[7];
                rf2_z[2] = RP2.cloud_REFERENCE_FRAME_descriptors[nn_indices[j]].vector[8];
                rf2_z[3] = 0;

                double angle_x = pcl::getAngle3D(rf1_x,rf2_x);
                double angle_y = (pcl::getAngle3D(rf1_y,rf2_y));
                double angle_z = (pcl::getAngle3D(rf1_z,rf2_z));


                angles_vector.push_back(std::max(angle_x,std::max(angle_y,angle_z)));


            }
            //cout << "Second THreshold Potential Matches: " << good_angles_accumulate_indices.size()<< endl;

            std::vector<double>::iterator result;
            result = std::min_element(angles_vector.begin(), angles_vector.end());
            //std::cout << "Max element at: " << std::distance(match_distance.begin(), result) << '\n';
            //std::cout << "Max element is: " << match_distance[std::distance(match_distance.begin(), result)] << '\n';
            int min_element_index = std::distance(angles_vector.begin(), result);

            pcl::Correspondence corr;
            corr.index_query = RP1.patch_descriptor_indices[i];// vulnerable
            corr.index_match = RP2.patch_descriptor_indices[nn_indices[min_element_index]];// vulnerable

            corrs.push_back(corr);

        }
    }
    end_shot2 = clock();
    cpu_time_used_shot2 = ((double) (end_shot2 - start_shot2)) / CLOCKS_PER_SEC;
    cout <<  "Time taken for Feature Descriptor Matching : " << (double)cpu_time_used_shot2 << "\n";

    //////////////////////////////////////////////////////////////////////////////////////
    /// \brief kdtree_LRF----------JUST REFERENCE FRAME descriptors
    //////////////////////////////////////////////////////////////////////////////////////



    cout << "No. of Reciprocal Correspondences : " << corrs.size() << endl;

    pcl::CorrespondencesConstPtr corrs_const_ptr = boost::make_shared< pcl::Correspondences >(corrs);

    pcl::Correspondences corr_shot;
    pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > Ransac_based_Rejection_shot;
    Ransac_based_Rejection_shot.setInputSource(RP1.cloud_keypoints.makeShared());
    Ransac_based_Rejection_shot.setInputTarget(RP2.cloud_keypoints.makeShared());
    Ransac_based_Rejection_shot.setInlierThreshold(0.03);
    Ransac_based_Rejection_shot.setInputCorrespondences(corrs_const_ptr);
    Ransac_based_Rejection_shot.getCorrespondences(corr_shot);




    cout << "True correspondences after RANSAC : " << corr_shot.size() << endl;





    //for normal point cloud matches visualization
    /////////////////////////////////////////////////////////////////////////
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);



    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(RP1.cloud_keypoints.makeShared(), 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (RP1.cloud_keypoints.makeShared(), single_color1, "sample cloud1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "sample cloud1");
    //viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    Eigen::Matrix4f t;
    t<<1,0,0,-0.4,
            0,1,0,-0.1,
            0,0,1,1,
            0,0,0,1;

    //cloudNext is my target cloud
    pcl::transformPointCloud(RP1.encoded3dkeypoints,RP1.encoded3dkeypoints,t);

    //int v2(1);
    //viewer->createViewPort (0.5,0.0,0.1,1.0,1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(RP1.encoded3dkeypoints.makeShared(), 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ> (RP1.encoded3dkeypoints.makeShared(), single_color2, "sample cloud2");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "sample cloud2");


    // see if these number of keypoints exist in the firstplace..or else it may give or show somethign wrong and illogical !!!
    pcl::Correspondences direct_corrs;
    for (int i = 0; i < 850; i = i+90)
    {
        pcl::Correspondence corr;
        corr.index_query = i;
        corr.index_match = i;
        direct_corrs.push_back(corr);
    }
    /*
for (int j = 0; j < 850; j = j+150)
{
    pcl::Correspondence corr;
    corr.index_query = j;
    corr.index_match = j;
    direct_corrs.push_back(corr);
}
*/

    cout << RP1.cloud_keypoints[500].x <<" " <<RP1.cloud_keypoints[500].y << " "<<RP1.cloud_keypoints[500].z;

    viewer->addCorrespondences<pcl::PointXYZ>(RP1.cloud_keypoints.makeShared(), RP1.encoded3dkeypoints.makeShared(), direct_corrs );




    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    /////////////////////////////////////////////////////////////////////////

    return 0;
}









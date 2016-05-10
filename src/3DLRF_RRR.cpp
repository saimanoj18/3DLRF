#include<3DLRF/3DLRF.h>


/************************************************************************
 *
 * Looping over the dataset point clouds starts
 *
 *
 * **********************************************************************/

#define MAXBUFSIZE  ((int) 1e6)


///To read the matrix
MatrixXf readMatrix(const char *filename)
{
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (! infile.eof())
    {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    MatrixXf result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
}

/************************************************************************
 *
 * Looping over the dataset point clouds ends
 *
 *
 * **********************************************************************/



int main(int argc, char **argv)
{

    threeDLRF RP1, RP2;

    ofstream ToFile;
    ToFile.open ("../results/3DLRF_12dim_0.001_ISS.txt", ios::out | ios::app);


    /*****************************************/
    if(argc < 4)
    {
        std::cerr << "Usage:" << std::endl;
        std::cerr << argv[0] << " [-a] model.ply scene.ply Tx file " << std::endl;
        std::cerr << "\t-a\tASCII output" << std::endl;
        return (1);
    }

    Eigen::Matrix4f T;
    T = readMatrix(argv[3]);
    //cout << T << endl;

    /*******************************************/



    /**********************Converting MESH to PointClouds**********************************/
    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFilePLY(argv[1], mesh);
    std::cerr << "Read cloud: " << std::endl;
    pcl::io::saveVTKFile ("temp2.vtk", mesh);

    // then use pcl_vtk2pcd

    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New ();
    reader->SetFileName ("temp2.vtk");
    reader->Update ();
    vtkSmartPointer<vtkPolyData> polydata = reader->GetOutput ();

    pcl::PointCloud<pcl::PointXYZ> cloud;
    vtkPolyDataToPointCloud (polydata, cloud);

    RP1.cloud = cloud;


    pcl::PolygonMesh mesh1;
    pcl::io::loadPolygonFilePLY(argv[2], mesh1);
    std::cerr << "Read cloud: " << std::endl;
    pcl::io::saveVTKFile ("temp3.vtk", mesh1);

    // then use pcl_vtk2pcd

    vtkSmartPointer<vtkPolyDataReader> reader1 = vtkSmartPointer<vtkPolyDataReader>::New ();
    reader1->SetFileName ("temp3.vtk");
    reader1->Update ();
    vtkSmartPointer<vtkPolyData> polydata1 = reader1->GetOutput ();

    pcl::PointCloud<pcl::PointXYZ> cloud1;
    vtkPolyDataToPointCloud (polydata1, cloud1);

    RP2.cloud = cloud1;
    /**********************Converting MESH to PointClouds**********************************/




    ///keypoint detection
    RP1.calculate_iss_keypoints(0.001);
    RP2.calculate_iss_keypoints(0.001);

    std::cout << "\nNo. of keypoints = "<< RP1.cloud_keypoints.size() << endl;

    clock_t start_, end_;
    double cpu_time_used_;
    start_ = clock();

    RP1.kdtree.setInputCloud(RP1.cloud.makeShared());// This is required for SUPER FAST
    RP2.kdtree.setInputCloud(RP2.cloud.makeShared());// THIS IS REQUIRED FOR SUPER FAST

    RP1.JUST_REFERENCE_FRAME_descriptors(0.06);
    RP2.JUST_REFERENCE_FRAME_descriptors(0.06);

    end_ = clock();
    cpu_time_used_ = ((double) (end_ - start_)) / CLOCKS_PER_SEC;
    std::cout << "Time taken for creating 3DLRF descriptors = " << (double)cpu_time_used_ << std::endl;
    ToFile <<"Time taken for creating 3D LRF descriptors + " << (double)cpu_time_used_ << endl;


    /************Reciprocal Correspondence Estimation :START **************/

    pcl::Correspondences corresp;


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

            corresp.push_back(corr);

        }
    }
    end_shot2 = clock();
    cpu_time_used_shot2 = ((double) (end_shot2 - start_shot2)) / CLOCKS_PER_SEC;
    cout <<  "Time taken for Feature Descriptor Matching : " << (double)cpu_time_used_shot2 << "\n";
    ToFile << "Time taken for matchign 3D LRF's : " << (double)cpu_time_used_shot2 << "\n";

    //////////////////////////////////////////////////////////////////////////////////////
    /// \brief kdtree_LRF----------JUST REFERENCE FRAME descriptors
    //////////////////////////////////////////////////////////////////////////////////////


    /************Reciprocal Correspondence Estimation : END**************/


    clock_t start1, end1;
    double cpu_time_used1;
    start1 = clock();

    pcl::CorrespondencesConstPtr correspond = boost::make_shared< pcl::Correspondences >(corresp);

    //RANSAC to find the true correspondences and remove the false positives
    pcl::Correspondences corr;
    pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > Ransac_based_Rejection;
    Ransac_based_Rejection.setInputSource(RP1.cloud_keypoints.makeShared());
    Ransac_based_Rejection.setInputTarget(RP2.cloud_keypoints.makeShared());
    Ransac_based_Rejection.setInlierThreshold(0.03);
    Ransac_based_Rejection.setInputCorrespondences(correspond);
    Ransac_based_Rejection.getCorrespondences(corr);

    end1 = clock();
    cpu_time_used1 = ((double) (end1 - start1)) / CLOCKS_PER_SEC;




    // To check the number of actual keypoints that are present both in the scene and the model
    // As described in the Evaluation Methodology in the paper
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(RP2.cloud_keypoints.makeShared());
    pcl::PointXYZ searchPoint;
    int actual_keypoints = 0;

    for(int i = 0; i < RP1.cloud_keypoints.size(); i++)
    {

        Eigen::Vector4f e_point1(RP1.cloud_keypoints[i].getVector4fMap());
        Eigen::Vector4f transformed_point(T*e_point1);

        //cout << T << endl;
        //cout << transformed_point << endl;

        searchPoint.x = transformed_point[0];
        searchPoint.y = transformed_point[1];
        searchPoint.z = transformed_point[2];

        int K = 1;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);

        if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            if ( 0.02 > sqrt(pointNKNSquaredDistance[0]))
            {
                actual_keypoints++;
            }
        }


    }


    cout << "Actual Keypoints : " << actual_keypoints << endl;


// To count the number of RANSAC Matches that are inline with the groundtruth correspondences
    int cnt=0;
    for (int i = 0; i < (int)corr.size(); i++)
    {
        pcl::PointXYZ point1 = RP1.cloud_keypoints[corr[i].index_query];
        pcl::PointXYZ point2 = RP2.cloud_keypoints[corr[i].index_match];

        Eigen::Vector4f e_point1(point1.getVector4fMap());
        Eigen::Vector4f e_point2(point2.getVector4fMap());

        Eigen::Vector4f transformed_point(T*e_point1);
        Eigen::Vector4f diff(e_point2 - transformed_point);

        if (diff.norm() < 0.05)
            cnt++;
    }

    ToFile << "RRR of 3D LRF's * " << ((float)cnt/(float)actual_keypoints)*100 << "\n \n";
    cout << "No. of RANSAC matches of 3DLRF descriptors : " << corr.size() << endl;
    cout << "Groundtruth matches of 3DLRF's' * " << cnt << endl;
    cout << "RRR of 3DLRF * : "<< ((float)cnt/(float)actual_keypoints)*100 << endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);

    //int v1(0);
    //viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(RP1.cloud_keypoints.makeShared(), 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (RP1.cloud_keypoints.makeShared(), single_color1, "sample cloud1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "sample cloud1");
    //viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    Eigen::Matrix4f t;
    t<<1,0,0,0.75,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1;

    //cloudNext is my target cloud
    pcl::transformPointCloud(RP2.cloud_keypoints,RP2.cloud_keypoints,t);

    //int v2(1);
    //viewer->createViewPort (0.5,0.0,0.1,1.0,1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(RP2.cloud_keypoints.makeShared(), 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ> (RP2.cloud_keypoints.makeShared(), single_color2, "sample cloud2");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "sample cloud2");


    viewer->addText ("PCA-SHOT", 10, 10, 30,0,1,0, "v1 text");
    viewer->addCorrespondences<pcl::PointXYZ>(RP1.cloud_keypoints.makeShared(), RP2.cloud_keypoints.makeShared(), corr, "correspondences");
    //viewer->addCorrespondences<pcl::PointXYZ>(RP1.cloud_keypoints.makeShared(), RP2.cloud_keypoints.makeShared(), corresp, "correspondences");


/*
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
*/

    return 0;



}



# include <Mfpfh/fpfh_low_dim.h>

#define MAXBUFSIZE  ((int) 1e6)

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




int main(int argc, char **argv)
{

    cbFPFH cb;

    ofstream myfile;
    myfile.open ("../results/FPFH_ISS_for_3DLRF.txt", ios::out | ios::app);


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


    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFilePLY(argv[1], mesh);
    std::cerr << "Read cloud: " << std::endl;
    pcl::io::saveVTKFile ("temp21112.vtk", mesh);
    // then use pcl_vtk2pcd
    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New ();
    reader->SetFileName ("temp21112.vtk");
    reader->Update ();
    vtkSmartPointer<vtkPolyData> polydata = reader->GetOutput ();
    pcl::PointCloud<pcl::PointXYZ> cloud;
    vtkPolyDataToPointCloud (polydata, cloud);
    cb.cloud1 = cloud;


    pcl::PolygonMesh mesh1;
    pcl::io::loadPolygonFilePLY(argv[2], mesh1);
    std::cerr << "Read cloud: " << std::endl;
    pcl::io::saveVTKFile ("temp31113.vtk", mesh1);
    // then use pcl_vtk2pcd
    vtkSmartPointer<vtkPolyDataReader> reader1 = vtkSmartPointer<vtkPolyDataReader>::New ();
    reader1->SetFileName ("temp31113.vtk");
    reader1->Update ();
    vtkSmartPointer<vtkPolyData> polydata1 = reader1->GetOutput ();
    pcl::PointCloud<pcl::PointXYZ> cloud1;
    vtkPolyDataToPointCloud (polydata1, cloud1);
    cb.cloud2 = cloud1;



    clock_t start_n, end_n;
    double cpu_time_used_n;
    start_n = clock();

    cb.calculate_normals (0.02);

    end_n = clock();
    cpu_time_used_n = ((double) (end_n - start_n)) / CLOCKS_PER_SEC;
    myfile << "Time taken for creating normals   " << (double)cpu_time_used_n << std::endl;

    cb.calculate_iss_keypoints_for_3DLRF(0.001);
    cb.get_keypoint_indices();

    std::cout << "\nNo. of keypoints : "<< cb.cloud1_keypoints.size() << endl;


    clock_t start_, end_;
    double cpu_time_used_;
    start_ = clock();

    cb.calculate_12_dim_FPFH(0.06);

    end_ = clock();
    cpu_time_used_ = ((double) (end_ - start_)) / CLOCKS_PER_SEC;
    std::cout << "Time taken for creating descriptors : " << (double)cpu_time_used_ << std::endl;
    myfile << "Time taken for creating FPFH descriptors # " << (double)cpu_time_used_ << "\n";

    //
    //for (int i = 0; i < 33; i++)
    //    cout << cb.cloud1_fpfh[25].histogram[i] <<" "<< endl;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cb.cloud2_keypoints.makeShared());
    pcl::PointXYZ searchPoint;
    int actual_keypoints = 0;

    for(int i = 0; i < cb.cloud1_keypoints.size(); i++)
    {

        Eigen::Vector4f e_point1(cb.cloud1_keypoints[i].getVector4fMap());
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
    cout << "Total Keypoints : "<< cb.cloud1_keypoints.size()<< endl;



    clock_t start_shot1, end_shot1;
    double cpu_time_used_shot1;
    start_shot1 = clock();


    // determine correspondences
    // This uses KdTree based NN search ..so is kind of Optimal :)..we are 10 times faster :)
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corr_est_shot;
    corr_est_shot.setInputSource(cb.cloud1_fpfh.makeShared()); // + setIndices(...)
    corr_est_shot.setInputTarget(cb.cloud2_fpfh.makeShared());
    pcl::Correspondences correspondeces_reciprocal_shot;
    //corr_est_shot.determineReciprocalCorrespondences(correspondeces_reciprocal_shot);
    corr_est_shot.determineCorrespondences(correspondeces_reciprocal_shot);
    //printf("%d\t reciprocal correspondences\n", (int)correspondeces_reciprocal.size());

    end_shot1 = clock();
    cpu_time_used_shot1 = ((double) (end_shot1 - start_shot1)) / CLOCKS_PER_SEC;

    cout <<  "Time taken for NN of FPFH : " << (double)cpu_time_used_shot1 << "\n";
    myfile <<  "Time taken for NN of FPFH : " << (double)cpu_time_used_shot1 << "\n";

    pcl::CorrespondencesConstPtr correspond_shot1 = boost::make_shared< pcl::Correspondences >(correspondeces_reciprocal_shot);

    pcl::Correspondences corr_shot1;
    pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > Ransac_based_Rejection_shot1;
    Ransac_based_Rejection_shot1.setInputSource(cb.cloud1_keypoints.makeShared());
    Ransac_based_Rejection_shot1.setInputTarget(cb.cloud2_keypoints.makeShared());
    Ransac_based_Rejection_shot1.setInlierThreshold(0.03);
    Ransac_based_Rejection_shot1.setInputCorrespondences(correspond_shot1);
    Ransac_based_Rejection_shot1.getCorrespondences(corr_shot1);

    int cnt=0;
    for (int i = 0; i < (int)corr_shot1.size(); i++)
    {
        pcl::PointXYZ point1 = cb.cloud1_keypoints[corr_shot1[i].index_query];
        pcl::PointXYZ point2 = cb.cloud2_keypoints[corr_shot1[i].index_match];

        Eigen::Vector4f e_point1(point1.getVector4fMap());

        Eigen::Vector4f e_point2(point2.getVector4fMap());

        Eigen::Vector4f transformed_point(T*e_point1);
        Eigen::Vector4f diff(e_point2 - transformed_point);

        if (diff.norm() < 0.05)
            cnt++;

    }





    //myfile << "Ground Truth Matches of SHOT * " << cnt << endl;
    cout << "Ground Truth Matches of FPFH * " << cnt << endl;
    cout << "Recogn. Rate of FPFH : "<< ((float)cnt/(float)actual_keypoints)*100 << endl;

    myfile << "Recogn. Rate of FPFH * "<< ((float)cnt/(float)actual_keypoints)*100 << endl;



    return 0;
}





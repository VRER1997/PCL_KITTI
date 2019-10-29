#include <bits/stdc++.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>

using namespace std;

void readKittiPclBinData(string &in_file, string& out_file)
{
    // load point cloud
    fstream input(in_file.c_str(), ios::in | ios::binary);
    if(!input.good()){
        cerr << "Could not read file: " << in_file << endl;
        exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::beg);
    pcl::PointCloud<pcl::PointXYZI>::Ptr points (new pcl::PointCloud<pcl::PointXYZI>);
    int i;
    for (i=0; input.good() && !input.eof(); i++) {
        pcl::PointXYZI point;
        input.read((char *) &point.x, 3*sizeof(float));
        input.read((char *) &point.intensity, sizeof(float));
        points->push_back(point);
    }
    input.close();
    pcl::io::savePCDFileBinary(out_file, *points);
}

void txt2pcd(string &in_file, string &out_file){

    fstream input(in_file);
    if(!input.good()){
        cerr << "Could not read file: " << in_file << endl;
        exit(EXIT_FAILURE);
    }
//    pcl::PointCloud::Ptr cloud(new pcl::PointCloud);
//    double x, y, z, r;
//    for(int i = 0; input.good() && !input.eof(); i++){
//        input >> x >> y >> z >> r;
//        cloud->points[i].x = x;
//        cloud->points[i].y = y;
//        cloud->points[i].z = z;
//    }
//    input.close();
//    pcl::io::savePCDFileBinary(out_file, cloud);
}

int main() {
    string base_path = "../velodyne/";
    string in_file = base_path + "000008.bin", out_file = "8.pcd";
    readKittiPclBinData(in_file, out_file);
//    txt2pcd(in_file, out_file);
    return 0;
}
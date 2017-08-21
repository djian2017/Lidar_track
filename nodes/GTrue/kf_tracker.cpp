//#include <ros/ros.h>
#include "Kalman.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>
//#if (CV_MAJOR_VERSION == 3)
#include "gencolors.cpp"
//#else
//#include <opencv2/contrib/contrib.hpp>
//#endif

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <boost/assert.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/algorithms/disjoint.hpp>
//#include <jsk_rviz_plugins/Pictogram.h>
//#include <jsk_rviz_plugins/PictogramArray.h>
#include "Cluster.h"
#include "KfLidarTracker.h"
#include <time.h>  
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <string>
//#include <opencv2/contrib/contrib.hpp>

typedef pcl::PointXYZ KittiPoint;
typedef pcl::PointCloud<KittiPoint> KittiPointCloud;

typedef boost::geometry::model::d2::point_xy<double> boost_point_xy;
typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > boost_polygon;

struct detect{
    int frame;
    int id;
    std::string obj_type;
    float truncation;
    int occlusion;
    float alpha;
    float x1;
    float y1;
    float x2;
    float y2;
    float h;
    float w;
    float l;
    geometry_msgs::Point pos;
    float ry;
    float score;
    ClusterPtr cluster;
};
std::vector<cv::Scalar> _colors;
bool pose_estimation = true;

std::vector< CTrack > tracks;

// KF parameter setting
float time_delta_ = 0.1;
float acceleration_noise_magnitude_ = 0.1;
float distance_threshold_ = 3;
float tracker_merging_threshold_ = 0.2;
size_t maximum_trace_length_ = 5;
size_t maximum_allowed_skipped_frames_ = 5;

int next_track_id_ = 0;
using namespace std;




double euclid_distance(const geometry_msgs::Point pos1,
//                              const pcl::PointXYZ pos2) {
                                  const geometry_msgs::Point pos2) {
  return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2) +
              pow(pos1.z - pos2.z, 2));

} 

std:: vector<detect> select_track(std::vector<detect> detection, int frame){
    std:: vector<detect> out_detect;
    for (unsigned int i=0; i<detection.size();i++){
        if (detection[i].frame==frame){
            out_detect.push_back(detection[i]);
        }
    }
    
    return out_detect;
}

std::vector<int> mycrop(const KittiPointCloud::Ptr in_cloud_ptr, const float h, const float w, const float l, const geometry_msgs::Point pos, const float ry)
{
    std::vector<int> out_ind;
    int pt_size = (in_cloud_ptr->points).size();
    
    float c = cos(ry);
    float s = sin(ry);
    Eigen:: Matrix3f P;
    P<< c,0,-1*s,
        0,1,0,
        s,0,c;
    
    // Lidar frame 2 camera frame
    Eigen:: Vector3f t;
    t(0) = -1*pos.y;
    t(1) = -1*pos.z;
    t(2) = pos.x;
    //std::cout<<pos.x<<" "<<pos.z<<std::endl;
    for (int i = 0; i< pt_size;i++)
    {   
        // Lidar frame 2 camera frame
        Eigen:: Vector3f pt,tp;
        pt(0)= -1*(in_cloud_ptr->points[i].y);
        pt(1)= -1*(in_cloud_ptr->points[i].z);
        pt(2)= in_cloud_ptr->points[i].x; 
        tp = P*(pt-t);
        if(l/-2.0<=tp(0) && tp(0)<=l/2.0 && -1*h<=tp(1)&& tp(1)<=0 && w/-2.0<=tp(2)&& tp(2)<=w/2.0)
        {
            out_ind.push_back(i);
        }
    }
    return out_ind;
}



void writecloud(std::vector<ClusterPtr>& clusters,std::vector<int> idx){
    pcl::PCDWriter writer;
    //pcl::PointCloud<pcl::PointXYZRGB> final_cloud;
    for(unsigned int i=0;i<idx.size();i++){
        std::vector<cv::Point2f> points;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cluster=clusters[idx[i]]->GetCloud();
        for (unsigned int i=0; i<current_cluster->points.size(); i++)
		{
			cv::Point2f pt;
			pt.x = current_cluster->points[i].x;
			pt.y = current_cluster->points[i].y;
			points.push_back(pt);
		}
        cv::RotatedRect box = minAreaRect(points);
        cout<<box.angle<<" "<<box.center.x<<" "<<box.center.y<<" "<<box.size.width<<" "<<box.size.height<<endl;
        
        stringstream ss;
        ss<< "cluster_"<<i<<".pcd";
        writer.write<pcl::PointXYZRGB> (ss.str(), *(clusters[idx[i]]->GetCloud()), false);
    }
     
}


void writecloud2(std::vector<ClusterPtr>& clusters){
    pcl::PCDWriter writer;
    pcl::PointCloud<pcl::PointXYZRGB> final_cloud;
    for(unsigned int i=0;i<clusters.size();i++){
        final_cloud+=*(clusters[i]->GetCloud());
    }
    stringstream ss;
    ss<< "cluster_all"<<".pcd";
    writer.write<pcl::PointXYZRGB> (ss.str(), final_cloud, false);
     
}

void writetracks(int framenum){
    
    std::ofstream file("tracks_lab2.txt", std::ios_base::app);
    for (unsigned int i=0;i<tracks.size();i++)
    {
        if (tracks[i].skipped_frames ==0)
        {
            jsk_recognition_msgs::BoundingBox tmp = tracks[i].cluster.GetBoundingBox();
            cv::RotatedRect box = tracks[i].cluster.GetRect();
            int truncation = 0;
            int occlusion = 1;
            float alpha = 0;
            float x1 = 0.0;
            float x2 = 0.0;
            float y1 = 0.0;
            float y2 = 0.0;
            cv::Point2f vertices[4];
            box.points(vertices);
            
            file<< framenum<<" "<<tracks[i].track_id<<" "<<"Car"<<" "<<truncation<<" "<<occlusion<<" "<<alpha;
            file<<" "<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" "<<tmp.dimensions.z<<" "<<tmp.dimensions.x<<" "<<tmp.dimensions.y<<" "<<tmp.pose.position.y<<" "<<tmp.pose.position.z+tmp.dimensions.z/2<<" "<<tmp.pose.position.x<<" ";
            file<<tracks[i].cluster.GetOrientationAngle()<<" ";
            for (int i=0; i<4;i++)
            {
                file<<vertices[i].x<<" "<<vertices[i].y<<" ";
            }
            file<<std::endl;
            /*
            std::cout<<vertices[1].x<<" "<< vertices[1].y<<std::endl;
            std::cout<<vertices[2].x<<" "<< vertices[2].y<<std::endl;
            std::cout<<vertices[3].x<<" "<< vertices[3].y<<std::endl;
            std::cout<<vertices[0].x<<" "<< vertices[0].y<<std::endl;
            std::cout<<"angle"<<box.angle<<std::endl;
            std::cout<<"center"<<box.center.x<<" "<<box.center.y<<std::endl;
            std::cout<<"size"<<box.size.width<<" "<<box.size.height<<std::endl<<std::endl;
            */
        }
        
    }
    file.close();
}

void showtracks(std::vector<CTrack>& tracks_in){
    for (unsigned i=0; i< tracks_in.size();i++){
        //std::cout<<tracks_in[i].GetCluster().GetCentroid().x<<" "<<tracks_in[i].GetCluster().GetCentroid().y<<std::endl;
        if (tracks_in[i].track_id==7 || tracks_in[i].track_id==9){
            cout<<tracks_in[i].track_id<<tracks_in[i].GetCluster().GetCentroid().x <<" "<<tracks_in[i].GetCluster().GetCentroid().y<<endl;
        }
        std::cout<<tracks_in[i].track_id<<std::endl;
    }
}


void obj_select(const std::vector<detect>& detect_in, const KittiPointCloud::Ptr in_cloud_ptr, std::vector<detect>& detect_out){
    
    //pcl::Cropbox<pcl::PointXYZ> crop;
    //crop.setInputCloud(in_cloud_ptr); 
    bool pose_estimation = true;
    //ClusterPtr cluster_tmp(new Cluster());
    int current_index = 0;
    for (unsigned i=0;i<detect_in.size();i++)
    {   
        if(detect_in[i].obj_type=="Car"||detect_in[i].obj_type=="Truck"||detect_in[i].obj_type=="Pedestrian"||detect_in[i].obj_type=="Van"||detect_in[i].obj_type=="Cyclist")
        //if(detect_in[i].obj_type=="Pedestrian")
        {
            detect tmp = detect_in[i];
            vector<int> out_ind = mycrop(in_cloud_ptr,tmp.h,tmp.w,tmp.l, tmp.pos, tmp.ry);
            if (out_ind.size() < 3)
            {
                std::cout<< i<<" is too few points"<<std::endl;
            }
            else
            {
                ClusterPtr cluster(new Cluster());
                cluster->SetCloud(in_cloud_ptr, out_ind, current_index,(int)_colors[i].val[0], (int)_colors[i].val[1], (int)_colors[i].val[2], "", pose_estimation);
                tmp.cluster = cluster;
                detect_out.push_back(tmp);
                current_index++;
            }
            out_ind.clear();
        }
    }
}

void CreatePolygonFromPoints(const geometry_msgs::Polygon& in_points, boost_polygon& out_polygon)
{
	std::vector< boost_point_xy > hull_detection_points;

	for (size_t k=0; k < in_points.points.size()/2; k++)
	{
		hull_detection_points.push_back(
					boost_point_xy(in_points.points[k].x,
							in_points.points[k].y)
					);
	}
	boost::geometry::assign_points(out_polygon, hull_detection_points);
}

void CheckTrackerMerge(size_t in_tracker_id, std::vector<CTrack>& in_trackers, std::vector<bool>& in_out_visited_trackers, std::vector<size_t>& out_merge_indices, double in_merge_threshold)
{   
    // Bug Fix by Jian Deng 2017/07/16
    if( std::find(out_merge_indices.begin(),out_merge_indices.end(),in_tracker_id) == out_merge_indices.end()){
        out_merge_indices.push_back(in_tracker_id);
    }
    
    
	for(size_t i=0; i< in_trackers.size(); i++)
	{
		if (i != in_tracker_id && !in_out_visited_trackers[i])
		{
			double distance =  sqrt( pow(in_trackers[in_tracker_id].GetCluster().GetCentroid().x - in_trackers[i].GetCluster().GetCentroid().x,2) +
										pow(in_trackers[in_tracker_id].GetCluster().GetCentroid().y - in_trackers[i].GetCluster().GetCentroid().y,2)
								);
			boost_polygon in_tracker_poly;
			CreatePolygonFromPoints(in_trackers[in_tracker_id].GetCluster().GetPolygon().polygon, in_tracker_poly);
			in_trackers[in_tracker_id].area = boost::geometry::area(in_tracker_poly);

			boost_polygon current_tracker_poly;
			CreatePolygonFromPoints(in_trackers[i].GetCluster().GetPolygon().polygon, current_tracker_poly);
			in_trackers[i].area = boost::geometry::area(current_tracker_poly);

			if (!boost::geometry::disjoint(in_tracker_poly, current_tracker_poly)
			//	|| distance <= in_merge_threshold)
            	&& distance <= in_merge_threshold)
			{   

                cout<< "Merge"<<in_trackers[in_tracker_id].track_id<<" " << in_trackers[i].track_id<<endl;
                cout<<in_trackers[in_tracker_id].GetCluster().GetCentroid().x <<" "<<in_trackers[i].GetCluster().GetCentroid().x<<endl;
                cout<<in_trackers[in_tracker_id].GetCluster().GetCentroid().y <<" "<<in_trackers[i].GetCluster().GetCentroid().y<<endl;
                cout<<distance<<endl;

				in_out_visited_trackers[i] = true;
				out_merge_indices.push_back(i);
				CheckTrackerMerge(i, in_trackers, in_out_visited_trackers, out_merge_indices, in_merge_threshold);
			}
		}
	}
}

void MergeTrackers(std::vector<CTrack>& in_trackers, std::vector<CTrack>& out_trackers, std::vector<size_t> in_merge_indices, const size_t& current_index, std::vector<bool>& in_out_merged_trackers)
{
	size_t oldest_life =0;
	size_t oldest_index = 0;
	double largest_area = 0.0f;
	size_t largest_index = 0;
	for (size_t i=0; i<in_merge_indices.size(); i++)
	{
		if (in_trackers[in_merge_indices[i]].life_span>= oldest_life)
		{
			oldest_life = in_trackers[in_merge_indices[i]].life_span;
			oldest_index = in_merge_indices[i];
		}
		if (in_trackers[in_merge_indices[i]].area>= largest_area)
        {
			largest_index = in_merge_indices[i];
        }
		in_out_merged_trackers[in_merge_indices[i]] = true;
	}

	out_trackers.push_back(in_trackers[oldest_index]);
	//out_trackers.back().cluster = in_trackers[largest_index].GetCluster();
}

void CheckAllTrackersForMerge(std::vector<CTrack>& out_trackers)
{
	//std::cout << "checkAllForMerge" << std::endl;
	std::vector<bool> visited_trackers(tracks.size(), false);
	std::vector<bool> merged_trackers(tracks.size(), false);
	size_t current_index=0;
	for (size_t i = 0; i< tracks.size(); i++)
	{
		if (!visited_trackers[i])
		{
			visited_trackers[i] = true;
			std::vector<size_t> merge_indices;
			CheckTrackerMerge(i, tracks, visited_trackers, merge_indices, tracker_merging_threshold_);
            //for (unsigned j=0; j<merge_indices.size();j++){
            //        std::cout<<merge_indices[j]<<" ";
           // }
            //std::cout<<endl;

			MergeTrackers(tracks, out_trackers, merge_indices, current_index++, merged_trackers);
            //showtracks(out_trackers);
            //std::cout<<endl;
            //showtracks(tracks);
            //std::cout<<out_trackers.size()<<std::endl;
		}
	}
    
	for(size_t i =0; i< tracks.size(); i++)
	{
		//check for clusters not merged, add them to the output
		if (!merged_trackers[i])
		{
			out_trackers.push_back(tracks[i]);
		}
	}

	//ClusterPtr cluster(new Cluster());
}



void Kfupdate(std::vector<detect>& det){
    size_t num_detections = det.size();
	size_t num_tracks = tracks.size();
	std::vector<int> track_assignments(num_tracks, -1);
	std::vector< std::vector<size_t> > track_assignments_vector(num_tracks);
    std::vector<size_t> detections_assignments;
    std::vector<float> track_distance(num_tracks,distance_threshold_);
    std::vector< CTrack > final_tracks;
    
    if (num_tracks == 0 ){
        for (unsigned int i = 0; i < det.size(); ++i)
		{
			tracks.push_back(CTrack(*(det[i].cluster),
									time_delta_,
									acceleration_noise_magnitude_,
									next_track_id_++)
							);
        }
    }
   else
        {
        for (size_t i = 0; i < num_detections; i++)
		{

			float current_distance_threshold = distance_threshold_;

			//detection polygon
            boost_polygon hull_detection_polygon;
            CreatePolygonFromPoints(det[i].cluster->GetPolygon().polygon, hull_detection_polygon);
			for (size_t j = 0; j < num_tracks; j++)
			{
				//float current_distance = tracks[j].CalculateDistance(cv::Point2f(in_cloud_cluster_array.clusters[i].centroid_point.point.x, in_cloud_cluster_array.clusters[i].centroid_point.point.y));
				float current_distance = sqrt(
												pow(tracks[j].GetCluster().GetCentroid().x - det[i].pos.x, 2) +
												pow(tracks[j].GetCluster().GetCentroid().y - det[i].pos.y, 2)
										);

				//tracker polygon
				boost_polygon hull_track_polygon;
				CreatePolygonFromPoints(tracks[j].GetCluster().GetPolygon().polygon, hull_track_polygon);
                

			//	std::cout<<tracks[j].track_id<<" "<<i<<" "<<current_distance <<endl;

				//if(current_distance < current_distance_threshold)
				//if (!boost::geometry::disjoint(hull_detection_polygon, hull_track_polygon)
				//	||  (current_distance < current_distance_threshold))
				if ((!boost::geometry::disjoint(hull_detection_polygon, hull_track_polygon)
					||  (current_distance < current_distance_threshold))&&(current_distance <track_distance[j]))

				{
	                //if (tracks[j].track_id==9||tracks[j].track_id==7){
	                       std::cout<<tracks[j].track_id<<" "<<i<<" "<<current_distance <<endl;

					//assign the closest detection or overlapping
					current_distance_threshold = current_distance;
					track_distance[j] = current_distance;
					track_assignments[j] = i;//assign detection i to track j
					track_assignments_vector[j].push_back(i);//add current detection as a match
					//detections_assignments.push_back(j);///////////////////////////////////////
					detections_assignments.push_back(i);
				}
			}

        }


       // showtracks(tracks);
        //check assignmets
        for (size_t i = 0; i< num_tracks; i++)
            {
                //std::cout<< track_assignments[i]<<std::endl;
                if (track_assignments[i]>=0) //if this track was assigned, update kalman filter, reset remaining life
                {
                	if (tracks[i].track_id==9||tracks[i].track_id==7){
                		std::cout<<tracks[i].track_id<<" "<<track_assignments[i] <<endl;
                	}

                	//keep oldest
                    tracks[i].skipped_frames = 0;
                    tracks[i].Update(*(det[track_assignments[i]].cluster),//*summed_cloud_cluster,
                                    true,
                                    maximum_trace_length_);
                    //detections_assignments.push_back(track_assignments[i]);
                }
                else				     // if not matched continue using predictions, and increase life
                {   
                    Cluster tmp; //empty cluster
                    tracks[i].Update(tmp, //empty cluster
									false, //not matched,
									maximum_trace_length_
                                    );
                    tracks[i].skipped_frames++;
                }
                tracks[i].life_span++;
            }
       // showtracks(tracks);
    // If track life is long, remove it.
        for (size_t i = 0; i < tracks.size(); i++)
            {
                if (tracks[i].skipped_frames > maximum_allowed_skipped_frames_)
                {
                    tracks.erase(tracks.begin() + i);
                    i--;
                }
            }
        // Search for unassigned detections and start new trackers.
        int una = 0;
        for (size_t i = 0; i < num_detections; ++i)
            {
                std::vector<size_t>::iterator it = find(detections_assignments.begin(), detections_assignments.end(), i);
                if (it == detections_assignments.end())//if detection not found in the already assigned ones, add new tracker
                {
                    tracks.push_back(CTrack(*(det[i].cluster),
                                        time_delta_,
										acceleration_noise_magnitude_,
										next_track_id_++)
                                    );
				if (next_track_id_ > 200)
					next_track_id_ = 0;
				una++;
                }
            }
		//std::cout << "Trackers added: " << una << std::endl;
       // showtracks(tracks);
        //std::cout<<std::endl;
		//finally check trackers among them
		CheckAllTrackersForMerge(final_tracks);

		tracks = final_tracks;
        //showtracks(tracks);
    }
}



int main(int argc, char **argv)
{
    string datafile = "/media/jian/E/uwaterloo/autocar/tracking/Kittistereo/data_tracking_velodyne/training/velodyne";
    string calfile = "/media/jian/E/uwaterloo/autocar/tracking/Kittistereo/data_tracking_velodyne/training/calib";
    string detfile = "/media/jian/E/uwaterloo/autocar/tracking/Kittistereo/data_tracking_velodyne/training/label_02";
    int dataset = 2;
    int framenum ;
    
    char setname[5];
    char frname[8];
    
    
    //pcl::PointCloud<pcl::PointXYZ>::Ptr current_sensor_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr removed_points_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr inlanes_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr nofloor_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr onlyfloor_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr diffnormals_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr clipped_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clustered_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    //jsk_recognition_msgs::BoundingBoxArray boundingbox_array;
    //jsk_recognition_msgs::PolygonArray polygon_array;
    //std::vector<geometry_msgs::Point> centroids_array; 
    std::vector<ClusterPtr> clusters;
    
    std::vector<detect> detect2;
    //std::vector<tracklets> tracks_ped;
    //std::vector<tracklets> tracks_cyc;
    
    //string setname;
    //std::sprintf(setname,0,"%4d%",dataset);
    //std::to_string(dataset);
    //sstm << "%4d" << dataset;
    
    
    #if (CV_MAJOR_VERSION == 3)
        generateColors(_colors, 100);
    #else
        cv::generateColors(_colors, 100);
    #endif
    
    
    std::sprintf(setname,"%04d",dataset);
    
    
    string s2(setname);
    
    
    string calname = calfile+"/"+s2+".txt";
    string detname = detfile+"/"+s2+".txt";
    
    std:: vector<detect> det;
    std:: vector<detect> detect_frame;
    
    std::ifstream file2(detname.c_str(), std::ios_base::in);
    //std::getline(myfile, calstring);
    detect tmp;
    while(file2){
        // pos is in Lidar frame
        file2>>tmp.frame>>tmp.id>>tmp.obj_type>>tmp.truncation>>tmp.occlusion>>tmp.alpha>>tmp.x1>>tmp.y1>>tmp.x2>>tmp.y2>>tmp.h>>tmp.w>>tmp.l>>tmp.pos.y>>tmp.pos.z>>tmp.pos.x>>tmp.ry;
        tmp.id = -1;
        tmp.pos.z = -tmp.pos.z;
        tmp.pos.y = -tmp.pos.y;
        det.push_back(tmp);
    }

    file2.close();
    clock_t start,end; 
    KittiPointCloud::Ptr cloud(new KittiPointCloud);
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");
    //viewer->addCoordinateSystem (10.0 ,0.0 ,0.0 ,2.0);
    viewer->initCameraParameters ();

    viewer->setCameraPosition(-319.84, -26.7078, 193.933, //position
    		-45.1396, -13.3388, 40.0938,    // FocalPoint
			0.488574, -0.151685, 0.859236 ); //View
    //cam.pos = {-45.1396, -13.3388, 40.0938};
    viewer->setCameraFieldOfView(9.1311*3.1415926/180);
    viewer->setCameraClipDistances(214.067, 585.395);
    viewer->setPosition(65, 52);
    viewer->setSize(1855, 1028);


    // 233 for case 002; 446 for case 001; 143 for case 003; 313 for case 004
    for (framenum = 0; framenum< 233; framenum++){
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        std::sprintf(frname,"%06d",framenum);
        string s3(frname);
        string filename = datafile+"/"+s2+"/"+s3+".bin";
        string outfile = datafile+"/tmp/"+s3+".png";
    
        cloud->clear();
        removed_points_cloud_ptr->clear();
        downsampled_cloud_ptr->clear();
        colored_clustered_cloud_ptr->clear();
        clipped_cloud_ptr->clear();
        diffnormals_cloud_ptr->clear();
        onlyfloor_cloud_ptr->clear();
        inlanes_cloud_ptr->clear();
        nofloor_cloud_ptr->clear();
        detect_frame.clear();
        detect2.clear();
        
        // Poin cloud files
        std::ifstream file(filename.c_str(), ios::in |ios::binary);
        float intensity;
        //std::ifstream file();
        if(file.good()){
            file.seekg(0, std::ios::beg);
        int i;
        for (i = 0; file.good() && !file.eof(); i++) {
            KittiPoint point;
            file.read((char *) &point.x, 3*sizeof(float));
            //file.read((char *) &point.intensity, sizeof(float));
            file.read((char *) &intensity, sizeof(float));
            if (point.x >0){
            	cloud->push_back(point);
            }

            }
        file.close();
        }
    
        viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
    
        //std::cout<<tracks[0].id<<std::endl<<tracks[0].ry<<std::endl<<tracks[1].id<<std::endl;
        
        detect_frame = select_track(det,framenum);
    
        //std::cout<<tracks_frame.size()<<std::endl;
    
    
        std::cout<< framenum<<std::endl;
        start = clock();
        
        std::cout<<"NO detect "<< detect_frame.size()<<std::endl;
        //obj_fusion(detect_frame, clusters, detect2);
        obj_select(detect_frame, cloud, detect2);
        end = clock();
        std::cout << "objfusion:" << float(end-start)/CLOCKS_PER_SEC << "s" << std::endl;
        std::cout<< "No filter out "<<detect2.size()<<std::endl;
    
        //std::cout<< tracks.size()<<std::endl;
        start = clock();
        Kfupdate(detect2);
        end = clock();
        std::cout << "Kfupdate:" << float(end-start)/CLOCKS_PER_SEC << "s" << std::endl;
        std::cout<< "No tracks "<<tracks.size()<<std::endl<<std::endl;
        
        for(unsigned int i =0; i<tracks.size();i++){
        	if (tracks[i].skipped_frames == 0){
        		jsk_recognition_msgs::BoundingBox bbox = tracks[i].GetCluster().GetBoundingBox();
        		        	Eigen::Vector3f t_pt;
        		        	t_pt<< bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z;
        		        	Eigen::Quaternionf quat(bbox.pose.orientation.w,bbox.pose.orientation.x,bbox.pose.orientation.y,bbox.pose.orientation.z);
        		        	//quat.w = ;
        		        	//quat.x = ;
        		        	//quat.y = bbox.pose.orientation.y;
        		        	//quat.z = bbox.pose.orientation.z;
        		        	//tf::quaternionMsgToTF(bbox.pose.orientation,quat);
        		        	double w = bbox.dimensions.x;
        		        	double h = bbox.dimensions.y;
        		        	double l = bbox.dimensions.z;
        		        	pcl::PointXYZ ctr = tracks[i].GetCluster().GetCentroid();
        		        	ctr.z += 2;
        		        	viewer->addCube(t_pt,quat,w,h,l,std::to_string(i));
        		        	viewer->addText3D(to_string(tracks[i].track_id),ctr,1,1,1,1,to_string(i+tracks.size()));
        	}
        }
        viewer->saveScreenshot(outfile);
        viewer->spinOnce(100);
       // writetracks(framenum);
    }
    
    return 0;
}

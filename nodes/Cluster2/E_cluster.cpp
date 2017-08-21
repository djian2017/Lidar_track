//#include <ros/ros.h>
#include "Kalman.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>

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
#include <pcl/filters/statistical_outlier_removal.h>

#include <boost/assert.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/algorithms/disjoint.hpp>
//#include <jsk_rviz_plugins/Pictogram.h>
//#include <jsk_rviz_plugins/PictogramArray.h>
#include "Cluster.h"

#include <time.h>  
#include <stdio.h>
#include <sstream>
#include <iostream>
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

std:: vector<detect> select_track(const std::vector<detect> detection, int frame){
    std:: vector<detect> out_detect;
    for (unsigned int i=0; i<detection.size();i++){
        if (detection[i].frame==frame){
            out_detect.push_back(detection[i]);
        }
    }
    
    return out_detect;
}

void removePointsUpTo(const KittiPointCloud::Ptr in_cloud_ptr, KittiPointCloud::Ptr out_cloud_ptr, double in_distance = 40)
{
	out_cloud_ptr->points.clear();
	for (unsigned int i=0; i<in_cloud_ptr->points.size(); i++)
	{
		float origin_distance = sqrt( pow(in_cloud_ptr->points[i].x,2) + pow(in_cloud_ptr->points[i].y,2) );
		if (origin_distance > in_distance)
		{
			out_cloud_ptr->points.push_back(in_cloud_ptr->points[i]);
		}
	}
}

void denoise(const KittiPointCloud::Ptr in_cloud_ptr, KittiPointCloud::Ptr out_cloud_ptr, float mean_K =50, float StdThresh = 1.0)
{   
          // Create the filtering object
    pcl::StatisticalOutlierRemoval<KittiPoint> sor;
    sor.setInputCloud (in_cloud_ptr);
    sor.setMeanK (mean_K);
    sor.setStddevMulThresh (StdThresh);
    sor.filter (*out_cloud_ptr);
}

void downsampleCloud(const KittiPointCloud::Ptr in_cloud_ptr, KittiPointCloud::Ptr out_cloud_ptr, float in_leaf_size=0.2)
{
	pcl::VoxelGrid<KittiPoint> sor;
	sor.setInputCloud(in_cloud_ptr);
	sor.setLeafSize((float)in_leaf_size, (float)in_leaf_size, (float)in_leaf_size);
	sor.filter(*out_cloud_ptr);
}

void removeFloor(const KittiPointCloud:: Ptr in_cloud, KittiPointCloud:: Ptr cloud_nofloor, KittiPointCloud:: Ptr out_onlyfloor_cloud_ptr, float in_floor_max_angle=0.1, float in_max_height=0.1)
{   
    // remove the floor
    pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

	seg.setOptimizeCoefficients (true);
	seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setAxis(Eigen::Vector3f(0,0,1));
	seg.setEpsAngle(in_floor_max_angle);

	seg.setDistanceThreshold (in_max_height);//floor distance
	seg.setOptimizeCoefficients(true);
	seg.setInputCloud(in_cloud);
	seg.segment(*inliers, *coefficients);
	if (inliers->indices.size () == 0)
	{
		std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
	}

	//REMOVE THE FLOOR FROM THE CLOUD
	pcl::ExtractIndices<KittiPoint> extract;
	extract.setInputCloud (in_cloud);
	extract.setIndices(inliers);
	extract.setNegative(true);//true removes the indices, false leaves only the indices
    extract.filter(*cloud_nofloor);
}

void differenceNormalsSegmentation(const KittiPointCloud::Ptr in_cloud_ptr, KittiPointCloud::Ptr out_cloud_ptr)
{
	float small_scale=0.5;
	float large_scale=2.0;
	float angle_threshold=0.5;
	pcl::search::Search<KittiPoint>::Ptr tree;
    
	if (in_cloud_ptr->isOrganized ())
	{
		tree.reset (new pcl::search::OrganizedNeighbor<KittiPoint> ());
	}
	else
	{
		tree.reset (new pcl::search::KdTree<KittiPoint> (false));
	}

	// Set the input pointcloud for the search tree
	tree->setInputCloud (in_cloud_ptr);

	pcl::NormalEstimationOMP<KittiPoint, pcl::PointNormal> normal_estimation;
	//pcl::gpu::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> normal_estimation;
	normal_estimation.setInputCloud (in_cloud_ptr);
	normal_estimation.setSearchMethod (tree);

	normal_estimation.setViewPoint (std::numeric_limits<float>::max (), std::numeric_limits<float>::max (), std::numeric_limits<float>::max ());

	pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale (new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale (new pcl::PointCloud<pcl::PointNormal>);

	normal_estimation.setRadiusSearch (small_scale);
	normal_estimation.compute (*normals_small_scale);

	normal_estimation.setRadiusSearch (large_scale);
	normal_estimation.compute (*normals_large_scale);

	pcl::PointCloud<pcl::PointNormal>::Ptr diffnormals_cloud (new pcl::PointCloud<pcl::PointNormal>);
	pcl::copyPointCloud<KittiPoint, pcl::PointNormal>(*in_cloud_ptr, *diffnormals_cloud);

	// Create DoN operator
	pcl::DifferenceOfNormalsEstimation<KittiPoint, pcl::PointNormal, pcl::PointNormal> diffnormals_estimator;
	diffnormals_estimator.setInputCloud (in_cloud_ptr);
	diffnormals_estimator.setNormalScaleLarge (normals_large_scale);
	diffnormals_estimator.setNormalScaleSmall (normals_small_scale);

	diffnormals_estimator.initCompute();

	diffnormals_estimator.computeFeature(*diffnormals_cloud);

	pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond (new pcl::ConditionOr<pcl::PointNormal>() );
	range_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (
			new pcl::FieldComparison<pcl::PointNormal> ("curvature", pcl::ComparisonOps::GT, angle_threshold) )
			);
	// Build the filter
	pcl::ConditionalRemoval<pcl::PointNormal> cond_removal;
	cond_removal.setCondition(range_cond);
	cond_removal.setInputCloud (diffnormals_cloud);

	pcl::PointCloud<pcl::PointNormal>::Ptr diffnormals_cloud_filtered (new pcl::PointCloud<pcl::PointNormal>);

	// Apply filter
	cond_removal.filter (*diffnormals_cloud_filtered);

	pcl::copyPointCloud<pcl::PointNormal, KittiPoint>(*diffnormals_cloud, *out_cloud_ptr);
}

std::vector<ClusterPtr> clusterAndColor(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud_ptr,
        //jsk_recognition_msgs::BoundingBoxArray& in_out_boundingbox_array,
		double in_max_cluster_distance=0.5)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    int cluster_size_min = 20;
    int cluster_size_max = 100000;
    
	//create 2d pc
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*in_cloud_ptr, *cloud_2d);
	//make it flat
	for (size_t i=0; i<cloud_2d->points.size(); i++)
	{
		cloud_2d->points[i].z = 0;
	}

	if (cloud_2d->points.size() > 0)
		tree->setInputCloud (cloud_2d);

	std::vector<pcl::PointIndices> cluster_indices;

	//perform clustering on 2d cloud
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (in_max_cluster_distance); //
	ec.setMinClusterSize (cluster_size_min);
	ec.setMaxClusterSize (cluster_size_max);
	ec.setSearchMethod(tree);
	ec.setInputCloud (cloud_2d);
	ec.extract (cluster_indices);
	//use indices on 3d cloud

	/*pcl::ConditionalEuclideanClustering<pcl::PointXYZ> cec (true);
	cec.setInputCloud (in_cloud_ptr);
	cec.setConditionFunction (&independentDistance);
	cec.setMinClusterSize (cluster_size_min);
	cec.setMaxClusterSize (cluster_size_max);
	cec.setClusterTolerance (_distance*2.0f);
	cec.segment (cluster_indices);*/

	/////////////////////////////////
	//---	3. Color clustered points
	/////////////////////////////////
	unsigned int k = 0;
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

	std::vector<ClusterPtr> clusters;
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);//coord + color cluster
	for (auto it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		ClusterPtr cluster(new Cluster());
		cluster->SetCloud(in_cloud_ptr, it->indices, k, (int)_colors[k].val[0], (int)_colors[k].val[1], (int)_colors[k].val[2], "", pose_estimation);
		clusters.push_back(cluster);

		k++;
	}
	//std::cout << "Clusters: " << k << std::endl;
	return clusters;

}

void checkClusterMerge(size_t in_cluster_id, std::vector<ClusterPtr>& in_clusters, std::vector<bool>& in_out_visited_clusters, std::vector<size_t>& out_merge_indices, double in_merge_threshold)
{
	//std::cout << "checkClusterMerge" << std::endl;
	pcl::PointXYZ point_a = in_clusters[in_cluster_id]->GetCentroid();
	for(size_t i=0; i< in_clusters.size(); i++)
	{
		if (i != in_cluster_id && !in_out_visited_clusters[i])
		{
			pcl::PointXYZ point_b = in_clusters[i]->GetCentroid();
			double distance = sqrt( pow(point_b.x - point_a.x,2) + pow(point_b.y - point_a.y,2) );
			if (distance <= in_merge_threshold)
			{
				in_out_visited_clusters[i] = true;
				out_merge_indices.push_back(i);
				//std::cout << "Merging " << in_cluster_id << " with " << i << " dist:" << distance << std::endl;
				checkClusterMerge(i, in_clusters, in_out_visited_clusters, out_merge_indices, in_merge_threshold);
			}
		}
	}
}

void mergeClusters(const std::vector<ClusterPtr>& in_clusters, std::vector<ClusterPtr>& out_clusters, std::vector<size_t> in_merge_indices, const size_t& current_index, std::vector<bool>& in_out_merged_clusters)
{
	//std::cout << "mergeClusters:" << in_merge_indices.size() << std::endl;
	pcl::PointCloud<pcl::PointXYZRGB> sum_cloud;
	pcl::PointCloud<pcl::PointXYZ> mono_cloud;
	ClusterPtr merged_cluster(new Cluster());
	for (size_t i=0; i<in_merge_indices.size(); i++)
	{
		sum_cloud += *(in_clusters[in_merge_indices[i]]->GetCloud());
		in_out_merged_clusters[in_merge_indices[i]] = true;
	}
	std::vector<int> indices(sum_cloud.points.size(), 0);
	for (size_t i=0; i<sum_cloud.points.size(); i++)
	{
		indices[i]=i;
	}

	if (sum_cloud.points.size() > 0)
	{
		pcl::copyPointCloud(sum_cloud, mono_cloud);
		//std::cout << "mergedClusters " << sum_cloud.points.size() << " mono:" << mono_cloud.points.size() << std::endl;
		//cluster->SetCloud(in_cloud_ptr, it->indices, _velodyne_header, k, (int)_colors[k].val[0], (int)_colors[k].val[1], (int)_colors[k].val[2], "", _pose_estimation);
		merged_cluster->SetCloud(mono_cloud.makeShared(), indices, current_index,(int)_colors[current_index].val[0], (int)_colors[current_index].val[1], (int)_colors[current_index].val[2], "", pose_estimation);
		out_clusters.push_back(merged_cluster);
	}
}

void checkAllForMerge(std::vector<ClusterPtr>& in_clusters, std::vector<ClusterPtr>& out_clusters, float in_merge_threshold)
{
	//std::cout << "checkAllForMerge" << std::endl;
	std::vector<bool> visited_clusters(in_clusters.size(), false);
	std::vector<bool> merged_clusters(in_clusters.size(), false);
	size_t current_index=0;
	for (size_t i = 0; i< in_clusters.size(); i++)
	{
		if (!visited_clusters[i])
		{
			visited_clusters[i] = true;
			std::vector<size_t> merge_indices;
			checkClusterMerge(i, in_clusters, visited_clusters, merge_indices, in_merge_threshold);
			mergeClusters(in_clusters, out_clusters, merge_indices, current_index++, merged_clusters);
		}
	}
	for(size_t i =0; i< in_clusters.size(); i++)
	{
		//check for clusters not merged, add them to the output
		if (!merged_clusters[i])
		{
			out_clusters.push_back(in_clusters[i]);
		}
	}

	//ClusterPtr cluster(new Cluster());
}

void segmentByDistance(const KittiPointCloud::Ptr in_cloud_ptr,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud_ptr,
        std::vector<ClusterPtr>& clusters_out)
		//jsk_recognition_msgs::BoundingBoxArray& in_out_boundingbox_array,
        //std::vector<geometry_msgs::Point>& centroids_array,    
	//	lidar_tracker::centroids& in_out_centroids,
	//	lidar_tracker::CloudClusterArray& in_out_clusters,
    //	jsk_recognition_msgs::PolygonArray& in_out_polygon_array)
	//	jsk_rviz_plugins::PictogramArray& in_out_pictogram_array)
{
	//cluster the pointcloud according to the distance of the points using different thresholds (not only one for the entire pc)
	//in this way, the points farther in the pc will also be clustered

	//0 => 0-15m d=0.5
	//1 => 15-30 d=1
	//2 => 30-45 d=1.6
	//3 => 45-60 d=2.1
	//4 => >60   d=2.6
	/*
	std::vector<KittiPointCloud::Ptr> cloud_segments_array(5);
    std::vector<double> clustering_distances = {15, 30, 45, 60};
    //std::vector<double> clustering_distances = {80};
    std::vector<double> clustering_thresholds = {0.5,1.1,1.6,2.1,2.6};
    //std::vector<double> clustering_thresholds = {0.5,0.7,0.9,1.1,1.3};
    double cluster_merge_threshold = 3;// 1.5;
    */

    std::vector<KittiPointCloud::Ptr> cloud_segments_array(2);
        //std::vector<double> clustering_distances = {15, 30, 45, 60};
        std::vector<double> clustering_distances = {80};
        std::vector<double> clustering_thresholds = {0.5,2.6};
        //std::vector<double> clustering_thresholds = {0.5,0.7,0.9,1.1,1.3};
        double cluster_merge_threshold = 3;// 1.5;


	for(unsigned int i=0; i<cloud_segments_array.size(); i++)
	{
		KittiPointCloud::Ptr tmp_cloud(new KittiPointCloud);
		cloud_segments_array[i] = tmp_cloud;
	}

	for (unsigned int i=0; i<in_cloud_ptr->points.size(); i++)
	{
		KittiPoint current_point;
		current_point.x = in_cloud_ptr->points[i].x;
		current_point.y = in_cloud_ptr->points[i].y;
		current_point.z = in_cloud_ptr->points[i].z;

		float origin_distance = sqrt( pow(current_point.x,2) + pow(current_point.y,2) );

		if 		(origin_distance < clustering_distances[0] )	{cloud_segments_array[0]->points.push_back (current_point);}
		else if(origin_distance < clustering_distances[1])		{cloud_segments_array[1]->points.push_back (current_point);}
		else if(origin_distance < clustering_distances[2])		{cloud_segments_array[2]->points.push_back (current_point);}
		else if(origin_distance < clustering_distances[3])		{cloud_segments_array[3]->points.push_back (current_point);}
		else													{cloud_segments_array[4]->points.push_back (current_point);}
	}

	std::vector <ClusterPtr> all_clusters;
	for(unsigned int i=0; i<cloud_segments_array.size(); i++)
	{
//#ifdef GPU_CLUSTERING
//		std::vector<ClusterPtr> local_clusters = clusterAndColorGpu(cloud_segments_array[i], out_cloud_ptr, in_out_boundingbox_array, in_out_centroids, _clustering_thresholds[i]);
//#else
		//std::vector<ClusterPtr> local_clusters = clusterAndColor(cloud_segments_array[i], out_cloud_ptr, in_out_boundingbox_array, in_out_centroids, _clustering_thresholds[i]);
        std::vector<ClusterPtr> local_clusters = clusterAndColor(cloud_segments_array[i], out_cloud_ptr, clustering_thresholds[i]);
//#endif
		all_clusters.insert(all_clusters.end(), local_clusters.begin(), local_clusters.end());
	}

	//Clusters can be merged or checked in here
	//....
	//check for mergable clusters
	std::vector<ClusterPtr> mid_clusters;
	std::vector<ClusterPtr> final_clusters;

	if (all_clusters.size() > 0)
		checkAllForMerge(all_clusters, mid_clusters, cluster_merge_threshold);
	else
		mid_clusters = all_clusters;

	if (mid_clusters.size() > 0)
			checkAllForMerge(mid_clusters, final_clusters, cluster_merge_threshold);
	else
		final_clusters = mid_clusters;

	//Get final PointCloud to be published
	//in_out_polygon_array.header = 1;
	//in_out_pictogram_array.header = _velodyne_header;
    
    //std::cout<< final_clusters.size()<<std::endl;
	for(unsigned int i=0; i<final_clusters.size(); i++)
	{
		*out_cloud_ptr = *out_cloud_ptr + *(final_clusters[i]->GetCloud());

		jsk_recognition_msgs::BoundingBox bounding_box = final_clusters[i]->GetBoundingBox();
		geometry_msgs::PolygonStamped polygon = final_clusters[i]->GetPolygon();
		//jsk_rviz_plugins::Pictogram pictogram_cluster;
		//pictogram_cluster.header = _velodyne_header;

		//PICTO
		//pictogram_cluster.mode = pictogram_cluster.STRING_MODE;
		//pictogram_cluster.pose.position.x = final_clusters[i]->GetMaxPoint().x;
		//pictogram_cluster.pose.position.y = final_clusters[i]->GetMaxPoint().y;
		//pictogram_cluster.pose.position.z = final_clusters[i]->GetMaxPoint().z;
		//tf::Quaternion quat(0.0, -0.7, 0.0, 0.7);
		//tf::quaternionTFToMsg(quat, pictogram_cluster.pose.orientation);
		//pictogram_cluster.size = 4;
		//std_msgs::ColorRGBA color;
		//color.a = 1; color.r = 1; color.g = 1; color.b = 1;
		//pictogram_cluster.color = color;
		//pictogram_cluster.character = std::to_string( i );
		//PICTO

		//pcl::PointXYZ min_point = final_clusters[i]->GetMinPoint();
		//pcl::PointXYZ max_point = final_clusters[i]->GetMaxPoint();
		pcl::PointXYZ center_point = final_clusters[i]->GetCentroid();
		geometry_msgs::Point centroid;
		centroid.x = center_point.x; centroid.y = center_point.y; centroid.z = center_point.z;
		//bounding_box.header = _velodyne_header;
        //bounding_box.header = 1;
		//polygon.header = _velodyne_header;
        //polygon.header = 1;

		if (	final_clusters[i]->IsValid()
				//&& bounding_box.dimensions.x >0 && bounding_box.dimensions.y >0 && bounding_box.dimensions.z > 0
				//&&	bounding_box.dimensions.x < _max_boundingbox_side && bounding_box.dimensions.y < _max_boundingbox_side
				)
		{
			clusters_out.push_back(final_clusters[i]);
            //in_out_boundingbox_array.boxes.push_back(bounding_box);
            //centroids_array.push_back(centroid);
			//in_out_centroids.points.push_back(centroid);
			//_visualization_marker.points.push_back(centroid);

			//in_out_polygon_array.polygons.push_back(polygon);
			//in_out_pictogram_array.pictograms.push_back(pictogram_cluster);

			//lidar_tracker::CloudCluster cloud_cluster;
			//final_clusters[i]->ToRosMessage(_velodyne_header, cloud_cluster);
			//in_out_clusters.clusters.push_back(cloud_cluster);
		}
	}

	//for(size_t i=0; i< in_out_polygon_array.polygons.size();i++)
	//{
	//	in_out_polygon_array.labels.push_back(i);
	//}

}


void clipCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, float in_min_height=-1.3, float in_max_height=0.5)
{
	out_cloud_ptr->points.clear();
	for (unsigned int i=0; i<in_cloud_ptr->points.size(); i++)
	{
		if (in_cloud_ptr->points[i].z >= in_min_height &&
				in_cloud_ptr->points[i].z <= in_max_height)
		{
			out_cloud_ptr->points.push_back(in_cloud_ptr->points[i]);
		}
	}
}


void obj_fusion(std::vector<detect>& detect_in, std::vector<ClusterPtr>& clusters_in, std::vector<detect>& detect_out){
    double threshold_min_dist = 5;
    jsk_recognition_msgs::BoundingBox bbox;
    detect tmp;
    
    std::vector<int> obj_indices;

  for (unsigned int i = 0; i < detect_in.size(); ++i) {
    unsigned int min_idx = 0;
    double min_distance = DBL_MAX;

    /* calculate each euclid distance between reprojected position and centroids
     */
    if (detect_in[i].obj_type=="Car"||detect_in[i].obj_type=="Truck"||detect_in[i].obj_type=="Pedestrian"||detect_in[i].obj_type=="Van"||detect_in[i].obj_type=="Cyclist"){

        for (unsigned int j = 0; j < clusters_in.size(); j++) {
        double distance =
            euclid_distance(detect_in[i].pos,
                            (clusters_in[j]->GetBoundingBox()).pose.position);

        /* Nearest centroid correspond to this reprojected object */
        if (distance < min_distance) {
            min_distance = distance;
            min_idx = j;
            }
        }
        //std::cout<< tracks_in[i].pos.z<<std::endl;
        //std::cout<< min_distance<<std::endl;
        //std::cout<< min_idx<<std::endl;
        if (min_distance < threshold_min_dist) {
            obj_indices.push_back(min_idx);
        
            
            tmp = detect_in[i];
            bbox = clusters_in[min_idx]->GetBoundingBox();
            //tmp.bbox = bbox;
            tmp.pos = bbox.pose.position;
            tmp.ry = clusters_in[min_idx]->GetOrientationAngle();
            tmp.cluster = clusters_in[min_idx];
            detect_out.push_back(tmp);
        }
        else {
        obj_indices.push_back(-1);
        }
    }        
    

  }
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
        //final_cloud+=*(clusters[i]->GetCloud());
    	std::cout<<clusters[i]->GetCentroid()<<endl;
    }
  //  stringstream ss;
  //  ss<< "cluster_all"<<".pcd";
  //  writer.write<pcl::PointXYZRGB> (ss.str(), final_cloud, false);
     
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr denoise_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
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

    // 233 for case 002
    //for (framenum = 40; framenum< 70; framenum++){
    framenum = 53;
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        std::sprintf(frname,"%06d",framenum);
        string s3(frname);
        string filename = datafile+"/"+s2+"/"+s3+".bin";
        
    
        cloud->clear();
        removed_points_cloud_ptr->clear();
        downsampled_cloud_ptr->clear();
        denoise_cloud_ptr->clear();
        colored_clustered_cloud_ptr->clear();
        clipped_cloud_ptr->clear();
        diffnormals_cloud_ptr->clear();
        onlyfloor_cloud_ptr->clear();
        inlanes_cloud_ptr->clear();
        nofloor_cloud_ptr->clear();
        detect_frame.clear();
        detect2.clear();
        clusters.clear();
        
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
    

    
        //std::cout<<tracks[0].id<<std::endl<<tracks[0].ry<<std::endl<<tracks[1].id<<std::endl;
        
        detect_frame = select_track(det,framenum);
    
        //std::cout<<tracks_frame.size()<<std::endl;
    
    
        std::cout<< framenum<<std::endl;
        start = clock();
        if (false)
        {
            removePointsUpTo(cloud, removed_points_cloud_ptr);
            downsampleCloud(removed_points_cloud_ptr, downsampled_cloud_ptr);
        }
        else
        {   
            downsampled_cloud_ptr = cloud;
        }

        denoise(downsampled_cloud_ptr, denoise_cloud_ptr);
        //downsampleCloud(removed_points_cloud_ptr, downsampled_cloud_ptr, _leaf_size);
    
        //pcl::io::savePCDFile ("test2.pcd", *downsampled_cloud_ptr);
        //end = clock();
        //std::cout << "downsampleCloud:" << float(end-start)/CLOCKS_PER_SEC << "ms" << std::endl;
        //clipCloud(downsampled_cloud_ptr, clipped_cloud_ptr, _clip_min_height, _clip_max_height);
    
        //start = clock();
        clipCloud(denoise_cloud_ptr, clipped_cloud_ptr);
        //end = clock();
        //std::cout << "clipCloud:" << clipped_cloud_ptr->points.size() << "time " << float(end-start)/CLOCKS_PER_SEC << "ms" << std::endl;
    
        inlanes_cloud_ptr = clipped_cloud_ptr;
    
        //removefloor(cloud_filtered, cloud_nofloor);
        //start = clock();
        removeFloor(inlanes_cloud_ptr, nofloor_cloud_ptr, onlyfloor_cloud_ptr);
        //end = clock();
        //std::cout << "removeFloor:" << float(end-start)/CLOCKS_PER_SEC << "ms" << std::endl;
    
        //start = clock();
        differenceNormalsSegmentation(nofloor_cloud_ptr, diffnormals_cloud_ptr);
        //end = clock();
        //std::cout << "differenceNormalsSegmentation:" << float(end-start)/CLOCKS_PER_SEC << "ms" << std::endl;
    
        //start = clock();
        segmentByDistance(diffnormals_cloud_ptr, colored_clustered_cloud_ptr, clusters);
        //end = clock();
        //std::cout << "segmentByDistance:" << float(end-start)/CLOCKS_PER_SEC << "ms" << std::endl;
    
        //writecloud(clusters,{18,27,55});
        //writecloud2(clusters);
        //start = clock();

        std::cout<<"NO detect "<< detect_frame.size()<<std::endl;
        obj_fusion(detect_frame, clusters, detect2);
        end = clock();
        std::cout << "objfusion:" << float(end-start)/CLOCKS_PER_SEC << "s" << std::endl;
        std::cout<< "No filter out "<<detect2.size()<<std::endl;

		/*
        //std::cout<< tracks.size()<<std::endl;
        start = clock();
        Kfupdate(detect2);
        end = clock();
        std::cout << "Kfupdate:" << float(end-start)/CLOCKS_PER_SEC << "s" << std::endl;
        std::cout<< "No tracks "<<tracks.size()<<std::endl<<std::endl;
        */
        
        viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
        for(unsigned int i =0; i<clusters.size();i++){
        //for(unsigned int i =0; i<detect2.size();i++){
        	jsk_recognition_msgs::BoundingBox bbox = clusters[i]->GetBoundingBox();
        	//jsk_recognition_msgs::BoundingBox bbox = detect2[i].cluster->GetBoundingBox();
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
               		        	pcl::PointXYZ ctr = clusters[i]->GetCentroid();
               		        	//pcl::PointXYZ ctr = detect2[i].cluster->GetCentroid();
               		        	ctr.z += 2;
               		        	viewer->addCube(t_pt,quat,w,h,l,std::to_string(i));
               		        	viewer->addText3D(to_string(clusters[i]->GetId()),ctr,1,1,1,1,to_string(i+clusters.size()));
               		        	//viewer->addText3D(to_string(detect2[i].cluster->GetId()),ctr,1,1,1,1,to_string(i+detect2.size()));
               		        	//viewer->addText3D(to_string(framenum),ctr,1,1,1,1,to_string(i+detect2.size()));

        }
        viewer->spinOnce(100);
    //    writetracks(framenum);
    //}
    
    
    //pcl::io::savePCDFile ("test.pcd", *nofloor_cloud_ptr);
    getchar();
    
    return 0;
}

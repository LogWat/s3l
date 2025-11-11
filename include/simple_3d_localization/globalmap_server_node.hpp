#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// #include <pcl_ros/point_cloud.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl/filters/voxel_grid.h>

namespace s3l::map
{

class GlobalmapServerNode : public rclcpp::Node {
public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    GlobalmapServerNode(const rclcpp::NodeOptions & options)
    : Node("globalmap_server", options) 
    {
        std::string globalmap_pcd = this->declare_parameter<std::string>("globalmap_pcd", "");
        globalmap_.reset(new PointCloudT);
        pcl::io::loadPCDFile(globalmap_pcd, *globalmap_);
        globalmap_->header.frame_id = "map";

        std::ifstream utm_file(globalmap_pcd + ".utm");
        if (utm_file.is_open() && this->declare_parameter<bool>("convert_utm_to_local", false)) {
            double utm_easting, utl_northing, altitude;
            utm_file >> utm_easting >> utl_northing >> altitude;
            for (auto& pt : globalmap_->points) {
                pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utl_northing, altitude);
            }
            RCLCPP_INFO(this->get_logger(), "Global map offset by UTM reference coordinates: (%f, %f, %f)",
                        utm_easting, utl_northing, altitude);
        }

        // downsample globalmap
        double downsample_resolution = this->declare_parameter<double>("downsample_resolution", 0.1);
        PointCloudT::Ptr filtered (new PointCloudT);
        std::shared_ptr<pcl::VoxelGrid<PointT>> voxel_grid_filter(new pcl::VoxelGrid<PointT>());
        voxel_grid_filter->setInputCloud(globalmap_);
        voxel_grid_filter->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
        voxel_grid_filter->filter(*filtered);
        globalmap_ = filtered;
        RCLCPP_INFO(this->get_logger(), "Global map loaded with %zu points", globalmap_->size());

        globalmap_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("globalmap", rclcpp::QoS(1).transient_local());
        globalmap_update_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/map_request/pointcloud", rclcpp::QoS(10),
            std::bind(&GlobalmapServerNode::globalmapUpdateCallback, this, std::placeholders::_1));
            
        globalmap_pub_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000),
            std::bind(&GlobalmapServerNode::publishGlobalMap, this));
    }

    GlobalmapServerNode(const rclcpp::NodeOptions & options, std::string & node_name);

private:
    void publishGlobalMap() {
        if (!globalmap_) {
            RCLCPP_WARN(this->get_logger(), "Global map is not set. Cannot publish.");
            return;
        }
        sensor_msgs::msg::PointCloud2::UniquePtr msg(new sensor_msgs::msg::PointCloud2);
        pcl::toROSMsg(*globalmap_, *msg);
        msg->header.frame_id = "map";
        globalmap_pub_->publish(std::move(msg));
    }

    void globalmapUpdateCallback(const std_msgs::msg::String::ConstSharedPtr& msg) {
        RCLCPP_INFO(this->get_logger(), "Global map update requested: %s", msg->data.c_str());
        std::string globalmap_pcd = msg->data;
        globalmap_.reset(new PointCloudT);
        pcl::io::loadPCDFile(globalmap_pcd, *globalmap_);
        globalmap_->header.frame_id = "map";
        RCLCPP_INFO(this->get_logger(), "Global map updated with %zu points", globalmap_->size());

        // downsample globalmap
        double downsample_resolution = this->declare_parameter<double>("downsample_resolution", 0.1);
        PointCloudT::Ptr filtered (new PointCloudT);
        std::shared_ptr<pcl::VoxelGrid<PointT>> voxel_grid_filter(new pcl::VoxelGrid<PointT>());
        voxel_grid_filter->setInputCloud(globalmap_);
        voxel_grid_filter->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
        voxel_grid_filter->filter(*filtered);
        globalmap_ = filtered;
        publishGlobalMap();
    }


    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr globalmap_update_sub_;

    PointCloudT::Ptr globalmap_;
    rclcpp::TimerBase::SharedPtr globalmap_pub_timer_;
};

} // namespace s3l::map

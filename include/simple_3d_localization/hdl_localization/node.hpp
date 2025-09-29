#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/filters/filter.h>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_eigen_kdl/tf2_eigen_kdl.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <std_srvs/srv/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

#include <simple_3d_localization/hdl_localization/pose_estimator.hpp>

#include <simple_3d_localization/type.hpp>
#include <simple_3d_localization/msg/scan_matching_status.hpp>
#include <simple_3d_localization/srv/set_global_map.hpp>
#include <simple_3d_localization/srv/query_global_localization.hpp>

namespace s3l::hdl_localization 
{

class LocalizationNode : public rclcpp::Node {

public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    LocalizationNode(const rclcpp::NodeOptions & options)
    : rclcpp::Node("s3l_hdl_localization", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        robot_odom_frame_id_ = this->declare_parameter<std::string>("robot_odom_frame_id", "odom");
        odom_child_frame_id_ = this->declare_parameter<std::string>("odom_child_frame_id", "base_link");
        scanmatch_frame_id_ = this->declare_parameter<std::string>("scanmatch_frame_id", "ndt_base_link");
        reg_method_ = this->declare_parameter<std::string>("registration_method", "ndt_omp"); // "ndt_cuda", "ndt_omp", "gicp", "vgicp"
        ndt_neighbor_search_method_ = this->declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");
        use_imu_initializer_ = this->declare_parameter<bool>("use_imu_initializer", false);
        imu_initialized_ = use_imu_initializer_ ? false : true;
        invert_acc_ = this->declare_parameter<bool>("invert_acc", false);
        invert_gyro_ = this->declare_parameter<bool>("invert_gyro", false);
        use_omp_ = this->declare_parameter<bool>("use_omp", true);
        downsample_leaf_size_ = this->declare_parameter<double>("downsample_leaf_size", 0.1);
        gicp_correspondence_randomness_ = this->declare_parameter<int>("gicp_correspondence_randomness", 20);
        gicp_max_correspondence_distance_ = this->declare_parameter<double>("gicp_max_correspondence_distance", 1.0);
        gicp_voxel_resolution_ = this->declare_parameter<double>("gicp_voxel_resolution", 1.0);
        num_threads_ = this->declare_parameter<int>("num_threads", 8);
        cool_time_duration_ = this->declare_parameter<double>("cool_time_duration", 0.5);
        use_mahalanobis_gating_ = this->declare_parameter<bool>("use_mahalanobis_gating", false);
        use_detail_gating_ = this->declare_parameter<bool>("use_detail_gating", false);
        mahalanobis_threshold_ = this->declare_parameter<double>("mahalanobis_threshold", 33.11);
        use_imu_ = this->declare_parameter<bool>("use_imu", true);
        use_odom_ = this->declare_parameter<bool>("odometry_based_prediction", false);
        imu_initialized_ = use_odom_ ? true : imu_initialized_;
        lio_odom_initialized_ = use_odom_ ? false : true;

        ndt_neighbor_search_radius_ = this->declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
        ndt_resolution_ = this->declare_parameter<double>("ndt_resolution", 1.0);

        use_global_localization_ = this->declare_parameter<bool>("use_global_localization", false);
        if (use_global_localization_) {
            RCLCPP_INFO(this->get_logger(), "Global localization is enabled.");
            RCLCPP_INFO(this->get_logger(), "Wait for global localization services");
            // TODO 実装
        }

        if (!imu_initialized_) {
            RCLCPP_INFO(this->get_logger(), "Waiting for IMU initialization...");
            // initial_g_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            //     "imu_init/gravity", rclcpp::QoS(1).transient_local(),
            //     std::bind(&LocalizationNode::initialGravityCallback, this, std::placeholders::_1));
            initial_ba_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
                "imu_init/accel_bias", rclcpp::QoS(1).transient_local(),
                std::bind(&LocalizationNode::initialAccelBiasCallback, this, std::placeholders::_1));
            initial_bg_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
                "imu_init/gyro_bias", rclcpp::QoS(1).transient_local(),
                std::bind(&LocalizationNode::initialGyroBiasCallback, this, std::placeholders::_1));
        } else if (imu_initialized_) {
            RCLCPP_INFO(this->get_logger(), "IMU initialization is skipped. Using provided parameters.");
        } else if (!use_imu_) {
            RCLCPP_WARN(this->get_logger(), "IMU is not used.");
        }

        // filter type (ukf, ekf)
        std::string filter_type_str = this->declare_parameter<std::string>("filter_type", "ukf");
        if (filter_type_str == "ukf") {
            RCLCPP_INFO(this->get_logger(), "Using Unscented Kalman Filter (UKF) for pose estimation.");
            filter_type_ = FilterType::UKF;
        } else if (filter_type_str == "ekf") {
            RCLCPP_INFO(this->get_logger(), "Using Extended Kalman Filter (EKF) for pose estimation.");
            filter_type_ = FilterType::EKF;
        } else {
            RCLCPP_WARN(this->get_logger(), "Invalid filter type: %s. Using UKF by default.", filter_type_str.c_str());
            filter_type_ = FilterType::UKF;
        }

        imu_buffer_.clear();
        lio_odom_buffer_.clear();

        // registation method (ndt_omp, ndt_cuda, gicp, vgicp) setup
        std::lock_guard<std::mutex> lock(reg_mutex_);
        registration_ = createRegistration();

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(
            std::shared_ptr<rclcpp::Node>(this, [](auto) {}));

        if (use_imu_) {
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                "imu/data", rclcpp::SensorDataQoS(),
                std::bind(&LocalizationNode::imuCallback, this, std::placeholders::_1));
        }
        if (use_odom_) {
            lio_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                "lio/odometry", rclcpp::SensorDataQoS(),
                std::bind(&LocalizationNode::lioOdomCallback, this, std::placeholders::_1));
        }
        points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "points_raw", rclcpp::SensorDataQoS(),
            std::bind(&LocalizationNode::pointsCallback, this, std::placeholders::_1));
        globalmap_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "global_map", rclcpp::SensorDataQoS(),
            std::bind(&LocalizationNode::globalMapCallback, this, std::placeholders::_1));
        initialpose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "initialpose", rclcpp::QoS(1),
            std::bind(&LocalizationNode::initialPoseCallback, this, std::placeholders::_1));
            
        aligned_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "aligned_points", rclcpp::SensorDataQoS());
        pose_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "odom", rclcpp::QoS(5));
        status_pub_ = this->create_publisher<simple_3d_localization::msg::ScanMatchingStatus>(
            "scan_matching_status", rclcpp::SensorDataQoS());

        // service
        relocalize_srv_ = this->create_service<std_srvs::srv::Empty>(
            "relocalize", std::bind(&LocalizationNode::relocalize, this, std::placeholders::_1, std::placeholders::_2));
        set_globalmap_client_ = this->create_client<simple_3d_localization::srv::SetGlobalMap>("set_global_map");
        query_global_localization_client_ = this->create_client<simple_3d_localization::srv::QueryGlobalLocalization>("query_global_localization");

        // downsampler setup
        RCLCPP_INFO(this->get_logger(), "Downsampler leaf size: %f", downsample_leaf_size_);
        std::shared_ptr<pcl::VoxelGrid<PointT>> downsampler(new pcl::VoxelGrid<PointT>());
        downsampler->setLeafSize(downsample_leaf_size_, downsample_leaf_size_, downsample_leaf_size_);
        downsampler_ = downsampler;

        // global localization setup
        relocalizing_ = false;

        // initialize pose estimator
        specify_init_pose_ = this->declare_parameter<bool>("specify_init_pose", false);
        if (specify_init_pose_) {
            RCLCPP_INFO(this->get_logger(), "Specify initial pose is enabled.");
            auto init_pose = this->declare_parameter<std::vector<float>>("init_pose", {0.0f, 0.0f, 0.0f});
            auto init_quat = this->declare_parameter<std::vector<float>>("init_quat", {0.0, 0.0, 0.0, 1.0});
            if (init_pose.size() != 3 || init_quat.size() != 4) {
                RCLCPP_ERROR(this->get_logger(), "Invalid initial pose or quaternion size. Expected 3 and 4 elements respectively.");
                return;
            }
            Eigen::Vector3f pos(init_pose[0], init_pose[1], init_pose[2]);
            Eigen::Quaternionf quat(init_quat[3], init_quat[0], init_quat[1], init_quat[2]);
            pose_estimator_ = std::make_unique<PoseEstimator>(
                registration_, pos, quat, use_odom_, filter_type_, cool_time_duration_
            );
            pose_estimator_->useMahalanobisGating(use_mahalanobis_gating_);
            pose_estimator_->useDetailGating(use_detail_gating_);
            pose_estimator_->setMahalanobisThreshold(mahalanobis_threshold_);
            est_initialized_ = true;
            if (imu_initialized_ && !use_odom_) {
                RCLCPP_INFO(this->get_logger(), "Initializing pose estimator with IMU parameters.");
                pose_estimator_->initializeWithBiasAndGravity(imu_gravity_, imu_accel_bias_, imu_gyro_bias_);
            }
        }
    }

    LocalizationNode(const rclcpp::NodeOptions & options, const std::string & node_name);


private:
 // ---------------------------------------------------------------------------------------------
    // helper functions

    void relocalize(
        const std_srvs::srv::Empty::Request::SharedPtr req,
        std_srvs::srv::Empty::Response::SharedPtr res) {
        (void)req; // 未使用の警告を抑制
        (void)res;
        if (last_scan_ == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "No last scan available for relocalization.");
            return;
        }
        
        relocalizing_ = true;
        PointCloudT::ConstPtr scan = last_scan_;

        // TODO: hdl global localization implementation
    }



    std::shared_ptr<pcl::Registration<PointT, PointT>> createRegistration() {
        if (reg_method_ == "ndt_omp") {
            RCLCPP_INFO(this->get_logger(), "NDT_OMP is selected");
            std::shared_ptr<pclomp::NormalDistributionsTransform<PointT, PointT>> ndt_omp(new pclomp::NormalDistributionsTransform<PointT, PointT>());
            
            ndt_omp->setTransformationEpsilon(0.01);
            ndt_omp->setResolution(ndt_resolution_);
            ndt_omp->setNumThreads(num_threads_);
            if (ndt_neighbor_search_method_ == "DIRECT1") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT1);
                RCLCPP_INFO(this->get_logger(), "Using DIRECT1 neighborhood search method");
            } else if (ndt_neighbor_search_method_ == "DIRECT7") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
                RCLCPP_INFO(this->get_logger(), "Using DIRECT7 neighborhood search method");
            } else if (ndt_neighbor_search_method_ == "KDTREE") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::KDTREE);
                RCLCPP_INFO(this->get_logger(), "Using KDTREE neighborhood search method");
            } else {
                RCLCPP_ERROR(this->get_logger(), "Unknown NDT neighbor search method: %s", ndt_neighbor_search_method_.c_str());
                return nullptr;
            }
            return ndt_omp;
        } else if (reg_method_ == "ndt_cuda") {
            RCLCPP_INFO(this->get_logger(), "NDT_CUDA is selected");
            std::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>());
            ndt->setResolution(ndt_resolution_);
            if (ndt_neighbor_search_method_ == "D2D") {
                ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
            } else if (ndt_neighbor_search_method_ == "P2D") {
                ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Unknown NDT distance mode: %s", reg_method_.c_str());
                return nullptr;
            }
            if (ndt_neighbor_search_method_ == "DIRECT1") {
                ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
            } else if (ndt_neighbor_search_method_ == "DIRECT7") {
                ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
            } else if (ndt_neighbor_search_method_ == "DIRECT_RADIUS") {
                ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius_);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Unknown NDT neighbor search method: %s", ndt_neighbor_search_method_.c_str());
                return nullptr;
            }
            return ndt;
        } else if (reg_method_ == "gicp" || reg_method_ == "vgicp") {
            std::string method = reg_method_ == "gicp" ? "GICP" : "VGICP";
            RCLCPP_INFO(this->get_logger(), "%s is selected", method.c_str());
            std::shared_ptr<small_gicp::RegistrationPCL<PointT, PointT>> gicp(new small_gicp::RegistrationPCL<PointT, PointT>());
            gicp->setNumThreads(num_threads_);
            gicp->setMaxCorrespondenceDistance(gicp_max_correspondence_distance_);
            gicp->setCorrespondenceRandomness(gicp_correspondence_randomness_);
            gicp->setVoxelResolution(gicp_voxel_resolution_);
            gicp->setRegistrationType(reg_method_ == "gicp" ? "GICP" : "VGICP");
            gicp->setMaximumIterations(200);
            return gicp;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unknown registration method: %s", reg_method_.c_str());
        }
        return nullptr;
    }

    
    void publishOdometry(const rclcpp::Time& stamp, const Eigen::Matrix4f& pose) {
        // 入力 pose 検証
        if (!pose.allFinite()) {
            RCLCPP_WARN(this->get_logger(), "publishOdometry: pose has NaN. Skipping TF/odom publish.");
            return;
        }
        Eigen::Quaterniond q(pose.block<3,3>(0,0).cast<double>());
        if (!std::isfinite(q.w()) || q.norm() < 1e-10) {
            RCLCPP_WARN(this->get_logger(), "publishOdometry: invalid quaternion. Skipping.");
            return;
        }
        geometry_msgs::msg::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
        map_wrt_frame.header.stamp = stamp;
        map_wrt_frame.header.frame_id = robot_odom_frame_id_;
        map_wrt_frame.child_frame_id = "map";
        try {
            geometry_msgs::msg::TransformStamped frame_wrt_odom = tf_buffer_.lookupTransform(
                robot_odom_frame_id_, odom_child_frame_id_, stamp, rclcpp::Duration::from_seconds(0.1));
            // Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();
            geometry_msgs::msg::TransformStamped map_wrt_odom;
            tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);
            tf2::Transform odom_wrt_map;
            tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
            odom_wrt_map = odom_wrt_map.inverse(); // convert to odom frame
            geometry_msgs::msg::TransformStamped odom_trans;
            odom_trans.transform = tf2::toMsg(odom_wrt_map);
            odom_trans.header.stamp = stamp;
            odom_trans.header.frame_id = "map";
            odom_trans.child_frame_id = odom_child_frame_id_;
            tf_broadcaster_->sendTransform(odom_trans);
        } catch (const tf2::TransformException & ex) {
            geometry_msgs::msg::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
            odom_trans.header.stamp = stamp;
            odom_trans.header.frame_id = "map";
            odom_trans.child_frame_id = odom_child_frame_id_;
            tf_broadcaster_->sendTransform(odom_trans);
        }

        nav_msgs::msg::Odometry::UniquePtr odom_msg(new nav_msgs::msg::Odometry());
        odom_msg->header.stamp = stamp;
        odom_msg->header.frame_id = "map";
        odom_msg->child_frame_id = odom_child_frame_id_;
        odom_msg->pose.pose = tf2::toMsg(Eigen::Isometry3d(pose.cast<double>()));
        odom_msg->twist.twist.linear.x = 0.0;
        odom_msg->twist.twist.linear.y = 0.0;
        odom_msg->twist.twist.linear.z = 0.0;

        pose_pub_->publish(std::move(odom_msg));
    }

    void publishScanMatchingStatus(const std_msgs::msg::Header& header, PointCloudT::ConstPtr aligned) {
        simple_3d_localization::msg::ScanMatchingStatus::UniquePtr status = std::make_unique<simple_3d_localization::msg::ScanMatchingStatus>();
        status->header = header;
        status->has_converged = registration_->hasConverged();
        status->matching_error = 0.0;
        const double max_correspondence_dist = registration_->getMaxCorrespondenceDistance();
        const double max_valid_point_dist = 25.0; // TODO: parameterize this value

        int num_inliers = 0, num_valid_points = 0;
        std::vector<int> k_indices;
        std::vector<float> k_sq_dists;

        for (int i = 0; i < (int)aligned->size(); i++) {
            const auto& pt = aligned->at(i);
            if (pt.getVector3fMap().norm() > max_valid_point_dist) continue; // skip outliers
            num_valid_points++;
            registration_->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
            if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
                status->matching_error += std::sqrt(k_sq_dists[0]);
                num_inliers++;
            }
        }

        status->matching_error /= std::max(1, num_inliers);
        status->inlier_fraction = static_cast<float>(num_inliers) / std::max(1, num_valid_points);
        status->relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration_->getFinalTransformation().cast<double>())).transform;

        if (tf_broadcaster_) {
            geometry_msgs::msg::TransformStamped ndt_tf = tf2::eigenToTransform(
                Eigen::Isometry3d(registration_->getFinalTransformation().cast<double>())
            );
            ndt_tf.header = header;
            ndt_tf.header.frame_id = "map";
            ndt_tf.child_frame_id = scanmatch_frame_id_;
            tf_broadcaster_->sendTransform(ndt_tf);
        }

        status->prediction_labels.reserve(2);
        status->prediction_errors.reserve(2);
        std::vector<double> errors(6, 0.0);

        if (pose_estimator_->wo_prediction_error()) {
            status->prediction_labels.push_back(std_msgs::msg::String{});
            status->prediction_labels.back().data = "without prediction error";
            status->prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator_->wo_prediction_error().value().cast<double>())).transform);
        }
        if (pose_estimator_->imu_prediction_error()) {
            status->prediction_labels.push_back(std_msgs::msg::String{});
            status->prediction_labels.back().data = "imu";
            status->prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator_->imu_prediction_error().value().cast<double>())).transform);
        }

        status_pub_->publish(std::move(status));
    }

    // callbacks -------------------------------------------------------------------------------------
    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
        imu_buffer_.push_back(msg);
    }

    void lioOdomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(lio_odom_mutex_);
        lio_odom_buffer_.push_back(msg);
        if (!lio_odom_initialized_) lio_odom_initialized_ = true;
    }

    void pointsCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        if(!globalmap_ || globalmap_->empty()) {
            RCLCPP_WARN(this->get_logger(), "Global map is not set yet. Ignoring point cloud.");
            return;
        }

        if (use_imu_ && !imu_initialized_) {
            RCLCPP_WARN(this->get_logger(), "IMU is not initialized yet. Ignoring point cloud.");
            return;
        }
        if (use_odom_ && !lio_odom_initialized_) {
            RCLCPP_WARN(this->get_logger(), "LIO odometry is not initialized yet. Ignoring point cloud.");
            return;
        }

        const auto & stamp = rclcpp::Time(msg->header.stamp.sec, msg->header.stamp.nanosec);
        PointCloudT::Ptr pcl_cloud(new PointCloudT);
        pcl::fromROSMsg(*msg, *pcl_cloud);

        if (pcl_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud. Ignoring.");
            return;
        }

        // transform pointcloud into odom_child_frame_id
        PointCloudT::Ptr cloud(new PointCloudT);
        try {
            geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform(
                odom_child_frame_id_, msg->header.frame_id, stamp, rclcpp::Duration::from_seconds(0.1));
            pcl_ros::transformPointCloud(*pcl_cloud, *cloud, transform);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "Failed to transform point cloud: %s", ex.what());
            return;
        }

        // downsample point cloud
        PointCloudT::Ptr downsampled_cloud(new PointCloudT);
        if (use_omp_) {
            downsampled_cloud = small_gicp::voxelgrid_sampling_omp(*cloud, downsample_leaf_size_);
        } else if (downsampler_) {
            downsampler_->setInputCloud(cloud);
            downsampler_->filter(*downsampled_cloud);
        } else {
            RCLCPP_WARN(this->get_logger(), "Downsampler is not set. Using original point cloud.");
            downsampled_cloud = cloud;
        }
        last_scan_ = downsampled_cloud;

        // predict ------------------------------------------------------
        if (use_odom_) {
            std::vector<nav_msgs::msg::Odometry::ConstSharedPtr> local_lio_odom;
            {
                std::lock_guard<std::mutex> lk(lio_odom_mutex_);
                local_lio_odom.swap(lio_odom_buffer_);
            }
            std::sort(local_lio_odom.begin(), local_lio_odom.end(), [](const auto & a, const auto & b) {
                rclcpp::Time ta(a->header.stamp.sec, a->header.stamp.nanosec);
                rclcpp::Time tb(b->header.stamp.sec, b->header.stamp.nanosec);
                return ta < tb;
            });
            std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
            if (!pose_estimator_) {
                RCLCPP_ERROR(this->get_logger(), "Waiting for initial pose input!");
                return;
            }
            for (const auto& odom_msg : local_lio_odom) {
                const auto odom_stamp = rclcpp::Time(odom_msg->header.stamp.sec, odom_msg->header.stamp.nanosec);
                if (odom_stamp > stamp) {
                    lio_odom_buffer_.push_back(odom_msg);
                    continue;
                }
                Eigen::Vector3f vel(odom_msg->twist.twist.linear.x, odom_msg->twist.twist.linear.y, odom_msg->twist.twist.linear.z);
                Eigen::Vector3f omega(odom_msg->twist.twist.angular.x, odom_msg->twist.twist.angular.y, odom_msg->twist.twist.angular.z);
                try {
                    pose_estimator_->predict(odom_stamp, vel, omega);
                } catch (const std::exception & e) {
                    RCLCPP_ERROR(this->get_logger(), "Pose prediction failed: %s", e.what());
                }
            }
        } else {
            std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> local_imu;
            {
                std::lock_guard<std::mutex> lk(imu_buffer_mutex_);
                local_imu.swap(imu_buffer_);
            }
            // 時系列になるように
            std::sort(local_imu.begin(), local_imu.end(), [](const auto & a, const auto & b) {
                rclcpp::Time ta(a->header.stamp.sec, a->header.stamp.nanosec);
                rclcpp::Time tb(b->header.stamp.sec, b->header.stamp.nanosec);
                return ta < tb;
            });
            std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
            if (!pose_estimator_) {
                RCLCPP_ERROR(this->get_logger(), "Waiting for initial pose input!");
                return;
            }
            for (const auto& imu_msg : local_imu) {
                const auto imu_stamp = rclcpp::Time(imu_msg->header.stamp.sec, imu_msg->header.stamp.nanosec);
                if (imu_stamp > stamp) {
                    imu_buffer_.push_back(imu_msg);
                    continue;
                }
                Eigen::Vector3f acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                Eigen::Vector3f gyro(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                if (invert_acc_) acc = -acc;
                if (invert_gyro_) gyro = -gyro;
                try {
                    pose_estimator_->predict(imu_stamp, acc, gyro);
                } catch (const std::exception & e) {
                    RCLCPP_ERROR(this->get_logger(), "Pose prediction failed: %s", e.what());
                }
            }
        }

        // correct ------------------------------------------------------
        auto aligned = pose_estimator_->correct(stamp, downsampled_cloud);

        // NAN guard
        Eigen::Matrix4f pose = pose_estimator_->matrix();
        if (!pose.allFinite()) {
            RCLCPP_ERROR(this->get_logger(), "Estimated pose has NaN. Resetting pose estimator.");
            return;
        }

        if (aligned_pub_->get_subscription_count() > 0) {
            sensor_msgs::msg::PointCloud2::UniquePtr aligned_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
            pcl::toROSMsg(*aligned, *aligned_msg);
            aligned_msg->header.frame_id = "map";
            aligned_msg->header.stamp = stamp;
            aligned_pub_->publish(std::move(aligned_msg));
        }
        if (status_pub_->get_subscription_count() > 0) {
            publishScanMatchingStatus(msg->header, aligned);
        }

        publishOdometry(msg->header.stamp, pose_estimator_->matrix());
    }

    void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr& msg) {
        RCLCPP_INFO(this->get_logger(), "Initial pose received!");
        std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
        const auto& p = msg->pose.pose.position;
        const auto& q = msg->pose.pose.orientation;
        pose_estimator_.reset(new PoseEstimator(
            registration_,
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z).normalized(),
            use_odom_,
            filter_type_,
            cool_time_duration_
        ));
        pose_estimator_->useMahalanobisGating(use_mahalanobis_gating_);
        pose_estimator_->useDetailGating(use_detail_gating_);
        pose_estimator_->setMahalanobisThreshold(mahalanobis_threshold_);
        est_initialized_ = true;
        if (imu_initialized_ && !use_odom_) {
            pose_estimator_->initializeWithBiasAndGravity(imu_gravity_, imu_accel_bias_, imu_gyro_bias_);
        }
    }

    void globalMapCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        RCLCPP_INFO(this->get_logger(), "Global map received!");
        std::lock_guard<std::mutex> lock(reg_mutex_);
        if (registration_ == nullptr) registration_ = createRegistration();

        pcl::PointCloud<PointT>::Ptr map(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*msg, *map);
        std::vector<int> dummy;
        pcl::removeNaNFromPointCloud(*map, *map, dummy);
        map->is_dense = true;
        map->height = 1;
        map->width = map->size();

        if (map->empty()) {
            RCLCPP_ERROR(this->get_logger(), "Received empty/invalid global map. Waiting next...");
            return;
        }

        globalmap_ = map;
        registration_->setInputTarget(globalmap_);

        if (use_global_localization_) {
            RCLCPP_INFO(this->get_logger(), "Global localization is enabled. Waiting for initial pose input.");
            simple_3d_localization::srv::SetGlobalMap::Request req;
            simple_3d_localization::srv::SetGlobalMap::Response res;
            req.global_map = *msg;
            if (set_globalmap_client_->wait_for_service(std::chrono::seconds(5))) {
                auto future = set_globalmap_client_->async_send_request(std::make_shared<simple_3d_localization::srv::SetGlobalMap::Request>(req));
                if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) != rclcpp::FutureReturnCode::SUCCESS) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to call SetGlobalMap service.");
                } else {
                    RCLCPP_INFO(this->get_logger(), "Global map set successfully.");
                }
            } else {
                RCLCPP_ERROR(this->get_logger(), "SetGlobalMap service is not available.");
            }
        }

        // subscriber shutdown
        globalmap_sub_.reset();
    }


    // void initialGravityCallback(const geometry_msgs::msg::Vector3::ConstSharedPtr& _msg) {
    //     // imu_gravity_ = Eigen::Vector3f(msg->x, msg->y, msg->z);
    //     imu_gravity_ = Eigen::Vector3f(0, 0, -9.80665); // TODO: imu_initializer(local) ⇔ this (global)の差をなんとかする
    //     has_init_g = true;
    //     checkImuInitRead();
    // }
    void initialAccelBiasCallback(const geometry_msgs::msg::Vector3::ConstSharedPtr& msg) {
        imu_accel_bias_ = Eigen::Vector3f(msg->x, msg->y, msg->z);
        has_init_ba = true;
        checkImuInitRead();
    }
    void initialGyroBiasCallback(const geometry_msgs::msg::Vector3::ConstSharedPtr& msg) {
        imu_gyro_bias_ = Eigen::Vector3f(msg->x, msg->y, msg->z);
        has_init_bg = true;
        checkImuInitRead();
    }
    void checkImuInitRead() {
        if (!imu_initialized_ && has_init_ba && has_init_bg) {
            imu_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "IMU initialized with provided parameters.");
            RCLCPP_INFO(this->get_logger(), "  Gravity: [%f, %f, %f]", imu_gravity_.x(), imu_gravity_.y(), imu_gravity_.z());
            RCLCPP_INFO(this->get_logger(), "  Accel Bias: [%f, %f, %f]", imu_accel_bias_.x(), imu_accel_bias_.y(), imu_accel_bias_.z());
            RCLCPP_INFO(this->get_logger(), "  Gyro Bias: [%f, %f, %f]", imu_gyro_bias_.x(), imu_gyro_bias_.y(), imu_gyro_bias_.z());

            // shutdown subscribers
            // initial_g_sub_.reset();
            initial_ba_sub_.reset();
            initial_bg_sub_.reset();

            if (est_initialized_) {
                RCLCPP_INFO(this->get_logger(), "Updating pose estimator with IMU parameters.");
                pose_estimator_->initializeWithBiasAndGravity(imu_gravity_, imu_accel_bias_, imu_gyro_bias_);
            }
        }
    }

    // variables ------------------------------------------------------------------------------------
    std::string robot_odom_frame_id_, odom_child_frame_id_, scanmatch_frame_id_;
    std::string reg_method_, ndt_neighbor_search_method_;
    double ndt_neighbor_search_radius_;
    double ndt_resolution_;
    int gicp_correspondence_randomness_;
    double gicp_max_correspondence_distance_;
    double gicp_voxel_resolution_;
    int num_threads_;

    bool use_imu_, use_odom_;

    bool use_imu_initializer_, imu_initialized_, lio_odom_initialized_;
    bool invert_acc_, invert_gyro_;
    bool use_omp_;
    bool specify_init_pose_;
    bool use_mahalanobis_gating_;
    bool use_detail_gating_;

    double mahalanobis_threshold_;

    FilterType filter_type_; // ekf, ukf

    // init pose params
    double cool_time_duration_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr lio_odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;

    // rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr initial_g_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr initial_ba_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr initial_bg_sub_;
    Eigen::Vector3f imu_gravity_{0, 0, -9.80665}, imu_accel_bias_{0, 0, 0}, imu_gyro_bias_{0, 0, 0};
    bool has_init_g{false}, has_init_ba{false}, has_init_bg{false};

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
    rclcpp::Publisher<simple_3d_localization::msg::ScanMatchingStatus>::SharedPtr status_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // 複数callbackで参照される可能性

    // imu input buffer
    std::mutex imu_buffer_mutex_;
    std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer_;

    // lio_odom buffer
    std::mutex lio_odom_mutex_;
    std::vector<nav_msgs::msg::Odometry::ConstSharedPtr> lio_odom_buffer_;

    // globalmap and registration method
    pcl::PointCloud<PointT>::Ptr globalmap_;
    pcl::Filter<PointT>::Ptr downsampler_;
    std::mutex reg_mutex_;
    std::shared_ptr<pcl::Registration<PointT, PointT>> registration_;
    double downsample_leaf_size_;

    // pose estimator
    std::mutex pose_estimator_mutex_;
    std::unique_ptr<PoseEstimator> pose_estimator_;
    bool est_initialized_{false};

    // global localization
    bool use_global_localization_;
    std::atomic_bool relocalizing_;

    pcl::PointCloud<PointT>::ConstPtr last_scan_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr relocalize_srv_;
    rclcpp::Client<simple_3d_localization::srv::SetGlobalMap>::SharedPtr set_globalmap_client_;
    rclcpp::Client<simple_3d_localization::srv::QueryGlobalLocalization>::SharedPtr query_global_localization_client_;
};
} // namespace s3l::hdl_localization

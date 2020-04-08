#include "RIEKFVisualizer.h"


using namespace ov_msckf;


RIEKFVisualizer::RIEKFVisualizer(ros::NodeHandle &nh, RIEKFManager* app) : _nh(nh), _app(app) {


    // Setup our transform broadcaster
    mTfBr = new tf::TransformBroadcaster();

    // Setup pose and path publisher
    pub_poseimu = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/ov_msckf/poseimu", 2);
    ROS_INFO("Publishing: %s", pub_poseimu.getTopic().c_str());
    pub_odomimu = nh.advertise<nav_msgs::Odometry>("/ov_msckf/odomimu", 2);
    ROS_INFO("Publishing: %s", pub_odomimu.getTopic().c_str());
    pub_pathimu = nh.advertise<nav_msgs::Path>("/ov_msckf/pathimu", 2);
    ROS_INFO("Publishing: %s", pub_pathimu.getTopic().c_str());

    // 3D points publishing
    pub_points_msckf = nh.advertise<sensor_msgs::PointCloud2>("/ov_msckf/points_msckf", 2);
    ROS_INFO("Publishing: %s", pub_points_msckf.getTopic().c_str());
    pub_points_slam = nh.advertise<sensor_msgs::PointCloud2>("/ov_msckf/points_slam", 2);
    ROS_INFO("Publishing: %s", pub_points_msckf.getTopic().c_str());
    pub_points_aruco = nh.advertise<sensor_msgs::PointCloud2>("/ov_msckf/points_aruco", 2);
    ROS_INFO("Publishing: %s", pub_points_aruco.getTopic().c_str());
    pub_points_sim = nh.advertise<sensor_msgs::PointCloud2>("/ov_msckf/points_sim", 2);
    ROS_INFO("Publishing: %s", pub_points_sim.getTopic().c_str());

    // Our tracking image
    pub_tracks = nh.advertise<sensor_msgs::Image>("/ov_msckf/trackhist", 2);
    ROS_INFO("Publishing: %s", pub_tracks.getTopic().c_str());

    // Groundtruth publishers
    pub_posegt = nh.advertise<geometry_msgs::PoseStamped>("/ov_msckf/posegt", 2);
    ROS_INFO("Publishing: %s", pub_posegt.getTopic().c_str());
    pub_pathgt = nh.advertise<nav_msgs::Path>("/ov_msckf/pathgt", 2);
    ROS_INFO("Publishing: %s", pub_pathgt.getTopic().c_str());

    // Load groundtruth if we have it
    if (nh.hasParam("path_gt")) {
        std::string path_to_gt;
        nh.param<std::string>("path_gt", path_to_gt, "");
        DatasetReader::load_gt_file(path_to_gt, gt_states);
    }

    // Load if we should save the total state to file
    nh.param<bool>("save_total_state", save_total_state, false);

    // If the file is not open, then open the file
    if(save_total_state) {

        // files we will open
        std::string filepath_est, filepath_std, filepath_gt;
        nh.param<std::string>("filepath_est", filepath_est, "state_estimate.txt");
        nh.param<std::string>("filepath_std", filepath_std, "state_deviation.txt");
        nh.param<std::string>("filepath_gt", filepath_gt, "state_groundtruth.txt");

        // If it exists, then delete it
        if(boost::filesystem::exists(filepath_est))
            boost::filesystem::remove(filepath_est);
        if(boost::filesystem::exists(filepath_std))
            boost::filesystem::remove(filepath_std);

        // Open the files
        of_state_est.open(filepath_est.c_str());
        of_state_std.open(filepath_std.c_str());
        of_state_est << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;
        of_state_std << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;

    }

}



void RIEKFVisualizer::visualize() {


    // publish current image
    publish_images();

    // Return if we have not inited
    if(!_app->intialized())
        return;

    // Save the start time of this dataset
    if(!start_time_set) {
        rT1 =  boost::posix_time::microsec_clock::local_time();
        start_time_set = true;
    }

    // publish state
    publish_state();

    // publish points
    publish_features();

    // Publish gt if we have it
    publish_groundtruth();

}



void RIEKFVisualizer::visualize_final() {


    // TODO: publish our calibration final results


    // Publish RMSE if we have it
    if(!gt_states.empty()) {
        ROS_INFO("\033[0;95mRMSE average: %.3f (deg) orientation\033[0m",summed_rmse_ori/summed_number);
        ROS_INFO("\033[0;95mRMSE average: %.3f (m) position\033[0m",summed_rmse_pos/summed_number);
    }

    // Print the total time
    rT2 =  boost::posix_time::microsec_clock::local_time();
    ROS_INFO("\033[0;95mTIME: %.3f seconds\033[0m",(rT2-rT1).total_microseconds()*1e-6);

}



void RIEKFVisualizer::publish_state() {

    // Get the current state
    State* state = _app->get_state();

    inekf::RobotState rstate = _app->filter_p_->getState();

    auto rquat = Eigen::Quaterniond(rstate.getRotation());

    // Create pose of IMU (note we use the bag time)
    geometry_msgs::PoseWithCovarianceStamped poseIinM;
    poseIinM.header.stamp = ros::Time(state->timestamp());
    poseIinM.header.seq = poses_seq_imu;
    poseIinM.header.frame_id = "global";

    poseIinM.pose.pose.orientation.x = rquat.x();
    poseIinM.pose.pose.orientation.y = rquat.y();
    poseIinM.pose.pose.orientation.z = rquat.z();
    poseIinM.pose.pose.orientation.w = rquat.w();
    poseIinM.pose.pose.position.x = rstate.getPosition()[0];
    poseIinM.pose.pose.position.y = rstate.getPosition()[1];
    poseIinM.pose.pose.position.z = rstate.getPosition()[2];

    // Finally set the covariance in the message (in the order position then orientation as per ros convention)
    std::vector<Type*> statevars;
    auto pos = rstate.getPosition();

    Vec* pos_vec = new Vec(3);
    pos_vec->update(pos);

    JPLQuat* quat_vec =  new JPLQuat();
    quat_vec->update(rquat.vec());

    statevars.push_back(pos_vec);
    statevars.push_back(quat_vec);

    Eigen::Matrix<double,6,6> covariance_posori = StateHelper::get_marginal_covariance(_app->get_state(),statevars);
    for(int r=0; r<6; r++) {
        for(int c=0; c<6; c++) {
            poseIinM.pose.covariance[6*r+c] = covariance_posori(r,c);
        }
    }
    pub_poseimu.publish(poseIinM);

    //=========================================================
    //=========================================================

    // Our odometry message (note we do not fill out angular velocities)
    nav_msgs::Odometry odomIinM;
    odomIinM.header = poseIinM.header;
    odomIinM.pose.pose = poseIinM.pose.pose;
    odomIinM.pose.covariance = poseIinM.pose.covariance;
    odomIinM.child_frame_id = "imu";
    odomIinM.twist.twist.angular.x = 0; // we do not estimate this...
    odomIinM.twist.twist.angular.y = 0; // we do not estimate this...
    odomIinM.twist.twist.angular.z = 0; // we do not estimate this...

    auto rvel = rstate.getVelocity();
    odomIinM.twist.twist.linear.x = rvel[0];
    odomIinM.twist.twist.linear.y = rvel[1];
    odomIinM.twist.twist.linear.z = rvel[2];

    // Velocity covariance (linear then angular)
    Vec* rvel_type = new Vec(3);
    rvel_type->update(rvel);

    statevars.clear();
    statevars.push_back(rvel_type);
    Eigen::Matrix<double,6,6> covariance_linang = INFINITY*Eigen::Matrix<double,6,6>::Identity();
    covariance_linang.block(0,0,3,3) = StateHelper::get_marginal_covariance(_app->get_state(),statevars);
    for(int r=0; r<6; r++) {
        for(int c=0; c<6; c++) {
            odomIinM.twist.covariance[6*r+c] = (std::isnan(covariance_linang(r,c))) ? 0 : covariance_linang(r,c);
        }
    }
    pub_odomimu.publish(odomIinM);


    //=========================================================
    //=========================================================

    // Append to our pose vector
    geometry_msgs::PoseStamped posetemp;
    posetemp.header = poseIinM.header;
    posetemp.pose = poseIinM.pose.pose;
    poses_imu.push_back(posetemp);

    // Create our path (imu)
    nav_msgs::Path arrIMU;
    arrIMU.header.stamp = ros::Time::now();
    arrIMU.header.seq = poses_seq_imu;
    arrIMU.header.frame_id = "global";
    arrIMU.poses = poses_imu;
    pub_pathimu.publish(arrIMU);

    // Move them forward in time
    poses_seq_imu++;

    // Publish our transform on TF
    // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
    // NOTE: a rotation from GtoI in JPL has the same xyzw as a ItoG Hamilton rotation
    tf::StampedTransform trans;
    trans.stamp_ = ros::Time::now();
    trans.frame_id_ = "global";
    trans.child_frame_id_ = "imu";
    tf::Quaternion quat(state->imu()->quat()(0),state->imu()->quat()(1),state->imu()->quat()(2),state->imu()->quat()(3));
    trans.setRotation(quat);
    tf::Vector3 orig(state->imu()->pos()(0),state->imu()->pos()(1),state->imu()->pos()(2));
    trans.setOrigin(orig);
    mTfBr->sendTransform(trans);

    // Loop through each camera calibration and publish it
    for(const auto &calib : state->get_calib_IMUtoCAMs()) {
        // need to flip the transform to the IMU frame
        Eigen::Vector4d q_ItoC = calib.second->quat();
        Eigen::Vector3d p_CinI = -calib.second->Rot().transpose()*calib.second->pos();
        // publish our transform on TF
        // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
        // NOTE: a rotation from ItoC in JPL has the same xyzw as a CtoI Hamilton rotation
        tf::StampedTransform trans;
        trans.stamp_ = ros::Time::now();
        trans.frame_id_ = "imu";
        trans.child_frame_id_ = "cam"+std::to_string(calib.first);
        tf::Quaternion quat(q_ItoC(0),q_ItoC(1),q_ItoC(2),q_ItoC(3));
        trans.setRotation(quat);
        tf::Vector3 orig(p_CinI(0),p_CinI(1),p_CinI(2));
        trans.setOrigin(orig);
        mTfBr->sendTransform(trans);
    }

    delete pos_vec;
    delete quat_vec;
    delete rvel_type;

}



void RIEKFVisualizer::publish_images() {

    // Check if we have subscribers
    if(pub_tracks.getNumSubscribers()==0)
        return;

    // Get our trackers
    TrackBase *trackFEATS = _app->get_track_feat();
    TrackBase *trackARUCO = _app->get_track_aruco();

    // Get our image of history tracks
    cv::Mat img_history;
    trackFEATS->display_history(img_history,255,255,0,255,255,255);
    if(trackARUCO != nullptr) {
        trackARUCO->display_history(img_history, 0, 255, 255, 255, 255, 255);
        trackARUCO->display_active(img_history, 0, 255, 255, 255, 255, 255);
    }

    // Create our message
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    sensor_msgs::ImagePtr exl_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

    // Publish
    pub_tracks.publish(exl_msg);

}




void RIEKFVisualizer::publish_features() {

    // Check if we have subscribers
    if(pub_points_msckf.getNumSubscribers()==0 && pub_points_slam.getNumSubscribers()==0 &&
       pub_points_aruco.getNumSubscribers()==0 && pub_points_sim.getNumSubscribers()==0)
        return;

    // Get our good features
    std::vector<Eigen::Vector3d> feats_msckf = _app->get_good_features_MSCKF();

    // Declare message and sizes
    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = "global";
    cloud.header.stamp = ros::Time::now();
    cloud.width  = 3*feats_msckf.size();
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = false; // there may be invalid points

    // Setup pointcloud fields
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1,"xyz");
    modifier.resize(3*feats_msckf.size());

    // Iterators
    sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");

    // Fill our iterators
    for(const auto &pt : feats_msckf) {
        *out_x = pt(0); ++out_x;
        *out_y = pt(1); ++out_y;
        *out_z = pt(2); ++out_z;
    }

    // Publish
    pub_points_msckf.publish(cloud);

    //====================================================================
    //====================================================================

    // Get our good features
    std::vector<Eigen::Vector3d> feats_slam = _app->get_features_SLAM();

    // Declare message and sizes
    sensor_msgs::PointCloud2 cloud_SLAM;
    cloud_SLAM.header.frame_id = "global";
    cloud_SLAM.header.stamp = ros::Time::now();
    cloud_SLAM.width  = 3*feats_slam.size();
    cloud_SLAM.height = 1;
    cloud_SLAM.is_bigendian = false;
    cloud_SLAM.is_dense = false; // there may be invalid points

    // Setup pointcloud fields
    sensor_msgs::PointCloud2Modifier modifier_SLAM(cloud_SLAM);
    modifier_SLAM.setPointCloud2FieldsByString(1,"xyz");
    modifier_SLAM.resize(3*feats_slam.size());

    // Iterators
    sensor_msgs::PointCloud2Iterator<float> out_x_SLAM(cloud_SLAM, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y_SLAM(cloud_SLAM, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z_SLAM(cloud_SLAM, "z");

    // Fill our iterators
    for(const auto &pt : feats_slam) {
        *out_x_SLAM = pt(0); ++out_x_SLAM;
        *out_y_SLAM = pt(1); ++out_y_SLAM;
        *out_z_SLAM = pt(2); ++out_z_SLAM;
    }

    // Publish
    pub_points_slam.publish(cloud_SLAM);

    //====================================================================
    //====================================================================

    // Get our good features
    std::vector<Eigen::Vector3d> feats_aruco = _app->get_features_ARUCO();

    // Declare message and sizes
    sensor_msgs::PointCloud2 cloud_ARUCO;
    cloud_ARUCO.header.frame_id = "global";
    cloud_ARUCO.header.stamp = ros::Time::now();
    cloud_ARUCO.width  = 3*feats_aruco.size();
    cloud_ARUCO.height = 1;
    cloud_ARUCO.is_bigendian = false;
    cloud_ARUCO.is_dense = false; // there may be invalid points

    // Setup pointcloud fields
    sensor_msgs::PointCloud2Modifier modifier_ARUCO(cloud_ARUCO);
    modifier_ARUCO.setPointCloud2FieldsByString(1,"xyz");
    modifier_ARUCO.resize(3*feats_aruco.size());

    // Iterators
    sensor_msgs::PointCloud2Iterator<float> out_x_ARUCO(cloud_ARUCO, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y_ARUCO(cloud_ARUCO, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z_ARUCO(cloud_ARUCO, "z");

    // Fill our iterators
    for(const auto &pt : feats_aruco) {
        *out_x_ARUCO = pt(0); ++out_x_ARUCO;
        *out_y_ARUCO = pt(1); ++out_y_ARUCO;
        *out_z_ARUCO = pt(2); ++out_z_ARUCO;
    }

    // Publish
    pub_points_aruco.publish(cloud_ARUCO);

}



void RIEKFVisualizer::publish_groundtruth() {

    // Our groundtruth state
    Eigen::Matrix<double,17,1> state_gt;

    // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
    if((gt_states.empty() || !DatasetReader::get_gt_state(_app->get_state()->timestamp(), state_gt, gt_states))) {
        return;
    }


    // Get the GT and system state state
    Eigen::Matrix<double,16,1> state_ekf = _app->get_state()->imu()->value();

    // Create pose of IMU
    geometry_msgs::PoseStamped poseIinM;
    poseIinM.header.stamp = ros::Time(_app->get_state()->timestamp());
    poseIinM.header.seq = poses_seq_gt;
    poseIinM.header.frame_id = "global";
    poseIinM.pose.orientation.x = state_gt(1,0);
    poseIinM.pose.orientation.y = state_gt(2,0);
    poseIinM.pose.orientation.z = state_gt(3,0);
    poseIinM.pose.orientation.w = state_gt(4,0);
    poseIinM.pose.position.x = state_gt(5,0);
    poseIinM.pose.position.y = state_gt(6,0);
    poseIinM.pose.position.z = state_gt(7,0);
    pub_posegt.publish(poseIinM);

    // Append to our pose vector
    poses_gt.push_back(poseIinM);

    // Create our path (imu)
    nav_msgs::Path arrIMU;
    arrIMU.header.stamp = ros::Time::now();
    arrIMU.header.seq = poses_seq_gt;
    arrIMU.header.frame_id = "global";
    arrIMU.poses = poses_gt;
    pub_pathgt.publish(arrIMU);

    // Move them forward in time
    poses_seq_gt++;

    // Publish our transform on TF
    tf::StampedTransform trans;
    trans.stamp_ = ros::Time::now();
    trans.frame_id_ = "global";
    trans.child_frame_id_ = "truth";
    tf::Quaternion quat(state_gt(1,0),state_gt(2,0),state_gt(3,0),state_gt(4,0));
    trans.setRotation(quat);
    tf::Vector3 orig(state_gt(5,0),state_gt(6,0),state_gt(7,0));
    trans.setOrigin(orig);
    mTfBr->sendTransform(trans);

    //==========================================================================
    //==========================================================================

    // Difference between positions
    double dx = state_ekf(4,0)-state_gt(5,0);
    double dy = state_ekf(5,0)-state_gt(6,0);
    double dz = state_ekf(6,0)-state_gt(7,0);
    double rmse_pos = std::sqrt(dx*dx+dy*dy+dz*dz);

    // Quaternion error
    Eigen::Matrix<double,4,1> quat_gt, quat_st, quat_diff;
    quat_gt << state_gt(1,0),state_gt(2,0),state_gt(3,0),state_gt(4,0);
    quat_st << state_ekf(0,0),state_ekf(1,0),state_ekf(2,0),state_ekf(3,0);
    quat_diff = quat_multiply(quat_st,Inv(quat_gt));
    double rmse_ori = (180/M_PI)*2*quat_diff.block(0,0,3,1).norm();


    //==========================================================================
    //==========================================================================

    // Get covariance of pose
        Eigen::Matrix<double,6,6> covariance = _app->get_state()->Cov().block(_app->get_state()->imu()->pose()->id(),_app->get_state()->imu()->pose()->id(),6,6);
    // Eigen::Matrix<double,6,6> covariance = _app->filter_p_->getState().block(_app->get_state()->imu()->pose()->id(),_app->get_state()->imu()->pose()->id(),6,6);

    // Calculate NEES values
    double ori_nees = 2*quat_diff.block(0,0,3,1).dot(covariance.block(0,0,3,3).inverse()*2*quat_diff.block(0,0,3,1));
    Eigen::Vector3d errpos = state_ekf.block(4,0,3,1)-state_gt.block(5,0,3,1);
    double pos_nees = errpos.transpose()*covariance.block(3,3,3,3).inverse()*errpos;

    //==========================================================================
    //==========================================================================

    // Update our average variables
    summed_rmse_ori += rmse_ori;
    summed_rmse_pos += rmse_pos;
    summed_nees_ori += ori_nees;
    summed_nees_pos += pos_nees;
    summed_number++;

    // Nice display for the user
    ROS_INFO("\033[0;95merror to gt => %.3f, %.3f (deg,m) | average error => %.3f, %.3f (deg,m) | called %d times \033[0m",rmse_ori,rmse_pos,summed_rmse_ori/summed_number,summed_rmse_pos/summed_number, (int)summed_number);
    ROS_INFO("\033[0;95mnees => %.1f, %.1f (ori,pos) | average nees = %.1f, %.1f (ori,pos) \033[0m",ori_nees,pos_nees,summed_nees_ori/summed_number,summed_nees_pos/summed_number);


    //==========================================================================
    //==========================================================================



}




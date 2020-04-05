#include <string>
#include <algorithm>
#include <memory>

#pragma once

#include "../../invariant-ekf/include/InEKF.h"
#include "../../invariant-ekf/include/RobotState.h"

#include <Eigen/StdVector>

#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "init/InertialInitializer.h"
#include "feat/FeatureRepresentation.h"
#include "types/Landmark.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"


using namespace ov_msckf;

class RIEKFManager
{


    public:


        /**
         * @brief Default constructor, will load all configuration variables
         * @param nh ROS node handler which we will load parameters from
         */
        RIEKFManager(ros::NodeHandle& nh);

        void initialize_filter(inekf::RobotState initial_state, inekf::NoiseParams noise_params);

        /**
         * @brief Feed function for inertial data
         * @param timestamp Time of the inertial measurement
         * @param wm Angular velocity
         * @param am Linear acceleration
         */
        void feed_measurement_imu(double timestamp, Eigen::Vector3d wm, Eigen::Vector3d am);


        /**
         * @brief Feed function for a single camera
         * @param timestamp Time that this image was collected
         * @param img0 Grayscale image
         * @param cam_id Unique id of what camera the image is from
         */
        void feed_measurement_monocular(double timestamp, cv::Mat& img0, size_t cam_id);


        /**
         * @brief Given a state, this will initialize our IMU state.
         * @param imustate State in the MSCKF ordering: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
         */
        void initialize_with_gt(Eigen::Matrix<double,17,1> imustate) {

            // Initialize the system
            state->imu()->set_value(imustate.block(1,0,16,1));
            state->set_timestamp(imustate(0,0));
            is_initialized_vio = true;

            // Print what we init'ed with
            ROS_INFO("\033[0;32m[INIT]: INITIALIZED FROM GROUNDTRUTH FILE!!!!!\033[0m");
            ROS_INFO("\033[0;32m[INIT]: orientation = %.4f, %.4f, %.4f, %.4f\033[0m",state->imu()->quat()(0),state->imu()->quat()(1),state->imu()->quat()(2),state->imu()->quat()(3));
            ROS_INFO("\033[0;32m[INIT]: bias gyro = %.4f, %.4f, %.4f\033[0m",state->imu()->bias_g()(0),state->imu()->bias_g()(1),state->imu()->bias_g()(2));
            ROS_INFO("\033[0;32m[INIT]: velocity = %.4f, %.4f, %.4f\033[0m",state->imu()->vel()(0),state->imu()->vel()(1),state->imu()->vel()(2));
            ROS_INFO("\033[0;32m[INIT]: bias accel = %.4f, %.4f, %.4f\033[0m",state->imu()->bias_a()(0),state->imu()->bias_a()(1),state->imu()->bias_a()(2));
            ROS_INFO("\033[0;32m[INIT]: position = %.4f, %.4f, %.4f\033[0m",state->imu()->pos()(0),state->imu()->pos()(1),state->imu()->pos()(2));

        }


        /// If we are initialized or not
        bool intialized() {
            return is_initialized_vio;
        }

        /// Accessor to get the current state
        State* get_state() {
            return state;
        }

        /// Get feature tracker
        TrackBase* get_track_feat() {
            return trackFEATS;
        }

        /// Get aruco feature tracker
        TrackBase* get_track_aruco() {
            return trackARUCO;
        }

        /// Returns 3d features used in the last update in global frame
        std::vector<Eigen::Vector3d> get_good_features_MSCKF() {
            return good_features_MSCKF;
        }

        /// Returns 3d SLAM features in the global frame
        std::vector<Eigen::Vector3d> get_features_SLAM() {
            std::vector<Eigen::Vector3d> slam_feats;
            for (auto &f : state->features_SLAM()){
                if((int)f.first <= state->options().max_aruco_features) continue;
                if(FeatureRepresentation::is_relative_representation(f.second->_feat_representation)) {
                    // Assert that we have an anchor pose for this feature
                    assert(f.second->_anchor_cam_id!=-1);
                    // Get calibration for our anchor camera
                    Eigen::Matrix<double, 3, 3> R_ItoC = state->get_calib_IMUtoCAM(f.second->_anchor_cam_id)->Rot();
                    Eigen::Matrix<double, 3, 1> p_IinC = state->get_calib_IMUtoCAM(f.second->_anchor_cam_id)->pos();
                    // Anchor pose orientation and position
                    Eigen::Matrix<double,3,3> R_GtoI = state->get_clone(f.second->_anchor_clone_timestamp)->Rot();
                    Eigen::Matrix<double,3,1> p_IinG = state->get_clone(f.second->_anchor_clone_timestamp)->pos();
                    // Feature in the global frame
                    slam_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose()*(f.second->get_xyz(false) - p_IinC) + p_IinG);
                } else {
                    slam_feats.push_back(f.second->get_xyz(false));
                }
            }
            return slam_feats;
        }

        /// Returns 3d ARUCO features in the global frame
        std::vector<Eigen::Vector3d> get_features_ARUCO() {
            std::vector<Eigen::Vector3d> aruco_feats;
            for (auto &f : state->features_SLAM()){
                if((int)f.first > state->options().max_aruco_features) continue;
                if(FeatureRepresentation::is_relative_representation(f.second->_feat_representation)) {
                    // Assert that we have an anchor pose for this feature
                    assert(f.second->_anchor_cam_id!=-1);
                    // Get calibration for our anchor camera
                    Eigen::Matrix<double, 3, 3> R_ItoC = state->get_calib_IMUtoCAM(f.second->_anchor_cam_id)->Rot();
                    Eigen::Matrix<double, 3, 1> p_IinC = state->get_calib_IMUtoCAM(f.second->_anchor_cam_id)->pos();
                    // Anchor pose orientation and position
                    Eigen::Matrix<double,3,3> R_GtoI = state->get_clone(f.second->_anchor_clone_timestamp)->Rot();
                    Eigen::Matrix<double,3,1> p_IinG = state->get_clone(f.second->_anchor_clone_timestamp)->pos();
                    // Feature in the global frame
                    aruco_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose()*(f.second->get_xyz(false) - p_IinC) + p_IinG);
                } else {
                    aruco_feats.push_back(f.second->get_xyz(false));
                }
            }
            return aruco_feats;
        }



    protected:


        /**
         * @brief This function will try to initialize the state.
         *
         * This should call on our initalizer and try to init the state.
         * In the future we should call the structure-from-motion code from here.
         * This function could also be repurposed to re-initialize the system after failure.         *
         * @return True if we have successfully initialized
         */
        bool try_to_initialize();


        /**
         * @brief This will do the propagation and feature updates to the state
         * @param timestamp The most recent timestamp we have tracked to
         */
        void do_feature_propagate_update(double timestamp);


        /// Our master state object :D
        State* state;

        /// Propagator of our state
        Propagator* propagator;

        /// Boolean if we should do stereo tracking or if false do binocular
        bool use_stereo = true;

        /// Our sparse feature tracker (klt or descriptor)
        TrackBase* trackFEATS = nullptr;

        /// Our aruoc tracker
        TrackBase* trackARUCO = nullptr;

        /// State initializer
        InertialInitializer* initializer;

        /// Boolean if we are initialized or not
        bool is_initialized_vio = false;

        /// Our MSCKF feature updater
        UpdaterMSCKF* updaterMSCKF;

        /// Our MSCKF feature updater
        UpdaterSLAM* updaterSLAM;

        /// Good features that where used in the last update
        std::vector<Eigen::Vector3d> good_features_MSCKF;

        // Timing variables
        boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6;

        // Track how much distance we have traveled
        double timelastupdate = -1;
        double distance = 0;

        // Start delay we should wait before inserting the first slam feature
        double dt_statupdelay;
        double startup_time = -1;

        // Camera intrinsics that we will load in
        std::map<size_t,bool> camera_fisheye;
        std::map<size_t,Eigen::VectorXd> camera_calib;
        std::map<size_t,std::pair<int,int>> camera_wh;

        inekf::RobotState state_;

        std::shared_ptr<inekf::InEKF> filter_p_;

        double t_prev_ {0};
        Eigen::Matrix<double,6,1> imu_measurement_prev_;
};

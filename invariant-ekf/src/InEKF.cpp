/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley <m.ross.hartley@gmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   InEKF.cpp
 *  @author Ross Hartley
 *  @brief  Source file for Invariant EKF
 *  @date   September 25, 2018
 **/

#include "InEKF.h"

namespace inekf {

using namespace std;

Eigen::MatrixXd J_r(Eigen::MatrixXd y){
    Eigen::MatrixXd I = Eigen::Matrix3d::Identity(y.rows(), y.rows());
    double norm = y.norm();
    double norm2 = norm * norm;
    double norm3 = norm2 * norm;
    Eigen::MatrixXd Sy = skew(y);

    Eigen::MatrixXd result;

    result = I - (1 - cos(norm)) * Sy/norm2 + (norm - sin(norm)) * Sy * Sy / norm3;

    return result;
}

void removeRowAndColumn(Eigen::MatrixXd& M, int index);

// ------------ Observation -------------
// Default constructor
Observation::Observation(Eigen::VectorXd& Y, Eigen::VectorXd& b, Eigen::MatrixXd& H, Eigen::MatrixXd& N, Eigen::MatrixXd& PI) :
    Y(Y), b(b), H(H), N(N), PI(PI) {}

// Check if empty
bool Observation::empty() { return Y.rows() == 0; }

ostream& operator<<(ostream& os, const Observation& o) {
    os << "---------- Observation ------------" << endl;
    os << "Y:\n" << o.Y << endl << endl;
    os << "b:\n" << o.b << endl << endl;
    os << "H:\n" << o.H << endl << endl;
    os << "N:\n" << o.N << endl << endl;
    os << "PI:\n" << o.PI << endl;
    os << "-----------------------------------";
    return os;
}

// ------------ InEKF -------------
// Default constructor
InEKF::InEKF() : g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()){}

// Constructor with noise params
InEKF::InEKF(NoiseParams params) : g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), noise_params_(params) {}

// Constructor with initial state
InEKF::InEKF(RobotState state) : g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), state_(state) {}

// Constructor with initial state and noise params
InEKF::InEKF(RobotState state, NoiseParams params) : g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), state_(state), noise_params_(params) {}

// Return robot's current state
RobotState InEKF::getState() {
    return state_;
}

// Sets the robot's current state
void InEKF::setState(RobotState state) {
    state_ = state;
}

// Return noise params
NoiseParams InEKF::getNoiseParams() {
    return noise_params_;
}

// Sets the filter's noise parameters
void InEKF::setNoiseParams(NoiseParams params) {
    noise_params_ = params;
}

// Return filter's prior (static) landmarks
mapIntVector3d InEKF::getPriorLandmarks() {
    return prior_landmarks_;
}

// Set the filter's prior (static) landmarks
void InEKF::setPriorLandmarks(const mapIntVector3d& prior_landmarks) {
    prior_landmarks_ = prior_landmarks;
}

// Return filter's estimated landmarks
map<int,int> InEKF::getEstimatedLandmarks() {
    return estimated_landmarks_;
}


// InEKF Propagation - Inertial Data
void InEKF::Propagate(const Eigen::Matrix<double,6,1>& m, double dt) {
    PropagateIMU(m, dt);

    PropagateCameras(m, dt);

    PropagateCovariance(m, dt);
}

// IMU Propagation: X_imu (+) e_I
void InEKF::PropagateIMU(const Eigen::Matrix<double,6,1>& m, double dt) {
    Eigen::Vector3d w = m.head(3) - state_.getGyroscopeBias();    // Angular Velocity
    Eigen::Vector3d a = m.tail(3) - state_.getAccelerometerBias(); // Linear Acceleration

    Eigen::MatrixXd X = state_.getX();
    Eigen::MatrixXd P = state_.getP();

    // Extract State
    Eigen::Matrix3d R = state_.getRotation();
    Eigen::Vector3d v = state_.getVelocity();
    Eigen::Vector3d p = state_.getPosition();

    // Strapdown IMU motion model
    Eigen::Vector3d phi = w*dt;
    Eigen::Matrix3d R_pred = R * Exp_SO3(phi); // TODO(lowmanj) RIght or left multiply?
    Eigen::Vector3d v_pred = v + (R*a + g_)*dt;
    Eigen::Vector3d p_pred = p + v*dt + 0.5*(R*a + g_)*dt*dt;

    // Set new state (bias has constant dynamics)
    state_.setRotation(R_pred);
    state_.setVelocity(v_pred);
    state_.setPosition(p_pred);
}

// Cam Propagation: C_i (+) e_theta
void InEKF::PropagateCameras(const Eigen::Matrix<double,6,1>& m, double dt) {
    Eigen::Vector3d w = m.head(3) - state_.getGyroscopeBias();
    Eigen::Vector3d a = m.tail(3) - state_.getAccelerometerBias();

    Eigen::Vector3d e_theta = w * dt;
    Eigen::Vector3d e_p = a * dt * dt;

    for (int ind=0; ind < state_.Cameras_.size(); ind++){
        auto cam_i = state_.Cameras_[ind];
        auto R_i = cam_i.getRotation();
        auto pos_i = cam_i.getPosition();

        Eigen::Matrix3d new_R_i = Exp_SO3(e_theta) * R_i; // TODO(lowmanj) RIght or left multiply?
        Eigen::Vector3d new_pos_i = Exp_SO3(e_theta) * pos_i + R_i * e_p; // TODO(lowmanj) + R*e_p or + J_r(m)*e_p?

        cam_i.setRotation(new_R_i);
        cam_i.setPosition(new_pos_i);

        state_.updateCameraEstimate(ind, cam_i);
    }

}


// Eigen::MatrixXd Adjoint_SEK3(const Eigen::MatrixXd& X) {
//     // Compute Adjoint(X) for X in SE_K(3)
//     int K = X.cols()-3;
//     Eigen::MatrixXd Adj = Eigen::MatrixXd::Zero(3+3*K, 3+3*K);
//     Eigen::Matrix3d R = X.block<3,3>(0,0);
//     Adj.block<3,3>(0,0) = R;
//     for (int i=0; i<K; ++i) {
//         Adj.block<3,3>(3+3*i,3+3*i) = R;
//         Adj.block<3,3>(3+3*i,0) = skew(X.block<3,1>(0,3+i))*R;
//     }
//     return Adj;
// }

void InEKF::PropagateCovariance(const Eigen::Matrix<double,6,1>& m, double dt){
    Eigen::Vector3d w = m.head(3) - state_.getGyroscopeBias();    // Angular Velocity
    Eigen::Vector3d a = m.tail(3) - state_.getAccelerometerBias(); // Linear Acceleration

    int dimP = state_.dimP();
    int dimTheta = state_.dimTheta();
    auto num_landmarks = state_.getNumberCameras();
    auto R = state_.getRotation();
    auto p = state_.getPosition();
    auto v = state_.getVelocity();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dimP, dimP);
    A.block<3, 3>(3, 0) = skew(g_);
    A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();

    A.block<3, 3>(0, dimP-dimTheta) = -R;
    A.block<3, 3>(3, dimP-dimTheta + 3) = -R;

    // A.block<3, 3>(3, 12) = -skew(v) * R;
    // A.block<3, 3>(6, 12) = -skew(p) * R;

    // TODO(lowmanj) Does there need to be a -skew(camera_pos)*R in here somewhere?

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dimP, dimP);
    Eigen::MatrixXd Phi = I + A * dt;

    Eigen::MatrixXd adj = Eigen::MatrixXd::Zero(dimP, dimP);
    adj.block<3, 3>(0, 0) = R;
    adj.block<3, 3>(3, 3) = R;
    adj.block<3, 3>(6, 6) = R;
    adj.block<3, 3>(9, 9) = R;
    adj.block<3, 3>(3, 0) = skew(v) * R;
    adj.block<3, 3>(6, 0) = skew(p) * R;

    for (int cam_index = 0; cam_index < num_landmarks; cam_index++){
        int diag_ind = 15 + 6*cam_index;
        assert(diag_ind < adj.rows());
        assert(diag_ind < adj.cols());

        auto cam = state_.getCameraEstimate(cam_index);

        adj.block<3, 3>(diag_ind, diag_ind) = cam.getRotation();
        adj.block<3, 3>(diag_ind+3, diag_ind+3) = cam.getRotation();
        adj.block<3, 3>(diag_ind+3, diag_ind) = skew(cam.getPosition()) * cam.getRotation();

    }

    // Eigen::MatrixXd B = Eigen::MatrixXd::Zero(6 + 3*num_landmarks, 6);
    // B.block<3, 3>(0, 0) = -J_r(-w);
    // B.block<3, 3>(3, 0) = -skew(v) * J_r(-w);
    // B.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    // Eigen::MatrixXd G = adj*B;

    Eigen::MatrixXd Qk = Eigen::MatrixXd::Zero(dimP,dimP);

    // Fill in top 15x15 of Qk with Q_dn
    Qk.block<3,3>(0,0) = noise_params_.getGyroscopeCov();
    Qk.block<3,3>(3,3) = noise_params_.getAccelerometerCov();

    Eigen::MatrixXd Qk_hat = adj * Qk * adj.transpose();

    Eigen::MatrixXd P_pred = Phi * state_.getP() * Phi.transpose() + Qk_hat;
    state_.setP(P_pred);

    return;
}


// Correct State: Right-Invariant Observation
void InEKF::Correct(const Observation& obs) {
    std::cout << "Computing Kalman Gain" << std::endl;

    // Compute Kalman Gain
    Eigen::MatrixXd P = state_.getP();
    Eigen::MatrixXd PHT = P * obs.H.transpose();
    Eigen::MatrixXd S = obs.H * PHT + obs.N;
    Eigen::MatrixXd K = PHT * S.inverse();

    std::cout << "Copying along diagonals" << std::endl;
    // Copy X along the diagonals if more than one measurement
    Eigen::MatrixXd BigX;
    state_.copyDiagX(obs.Y.rows()/state_.dimX(), BigX);
    // Compute correction terms
    Eigen::MatrixXd Z = BigX*obs.Y - obs.b;
    Eigen::VectorXd delta = K*obs.PI*Z;
    Eigen::MatrixXd dX = Exp_SEK3(delta.segment(0,delta.rows()-state_.dimTheta()));
    Eigen::VectorXd dTheta = delta.segment(delta.rows()-state_.dimTheta(), state_.dimTheta());

    std::cout << "Updating state" << std::endl;
    // Update state
    Eigen::MatrixXd X_new = dX*state_.getX(); // Right-Invariant Update
    Eigen::VectorXd Theta_new = state_.getTheta() + dTheta;
    state_.setX(X_new);
    state_.setTheta(Theta_new);

    std::cout << "Updating covariance" << std::endl;
    // Update Covariance
    Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(state_.dimP(),state_.dimP()) - K*obs.H;
    Eigen::MatrixXd P_new = IKH * P * IKH.transpose() + K*obs.N*K.transpose(); // Joseph update form

    state_.setP(P_new);
}

// Create Observation from vector of landmark measurements
void InEKF::CorrectFeatures(std::vector<ov_core::Feature*> featsup_MSCKF) {

    Eigen::VectorXd Y;
    Eigen::VectorXd b;
    Eigen::MatrixXd N;
    Eigen::MatrixXd PI;

    double dimP = state_.dimP();
    double n_cams = state_.getNumberCameras();
    double H_num_cols = 9*n_cams + 15;

    Eigen::MatrixXd H;
    Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(dimP, dimP);
    Eigen::MatrixXd Hf = Eigen::MatrixXd::Zero(dimP, dimP);
    std::vector<CameraPose> cameras = state_.getCameras();

    int cam_ind = 0;
    for (auto feature_p: featsup_MSCKF){
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 6);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 6);

        A.block<3, 3>(0, 0) = - camera.getRotation() * skew(camera.getPosition());

        H.block<3, 6>(3*cam_ind, 15 + 9*cam_ind) = A;
        H.block<3, 6>(3*cam_ind, 15 + 9*cam_ind) = B;

        cam_ind ++;
    }

    // std::cout << "Correcting state using stacked observations" << std::endl;
    // // Correct state using stacked observation
    // Observation obs(Y,b,H,N,PI);
    // if (!obs.empty()) {
    //     this->Correct(obs);
    // }

    // // We don't need to augment state with landmarks because this is
    // // handles by the MSCKF
    // std::cout << "Updating estimated landmarks" << std::endl;
    // int startIndex = state_.getX().rows();
    // for (vectorLandmarksIterator it=new_landmarks.begin(); it!=new_landmarks.end(); ++it) {
    //     startIndex++;
    //     estimated_landmarks_.insert(pair<int,int> (it->id, startIndex));
    // }

    // std::cout << "Augmenting state with new landmarks" << std::endl;
    // // Augment state with newly detected landmarks
    // if (new_landmarks.size() > 0) {
    //     Eigen::MatrixXd X_aug = state_.getX();
    //     Eigen::MatrixXd P_aug = state_.getP();
    //     Eigen::Vector3d p = state_.getPosition();
    //     for (vectorLandmarksIterator it=new_landmarks.begin(); it!=new_landmarks.end(); ++it) {
    //         // Initialize new landmark mean
    //         int startIndex = X_aug.rows();
    //         X_aug.conservativeResize(startIndex+1, startIndex+1);
    //         X_aug.block(startIndex,0,1,startIndex) = Eigen::MatrixXd::Zero(1,startIndex);
    //         X_aug.block(0,startIndex,startIndex,1) = Eigen::MatrixXd::Zero(startIndex,1);
    //         X_aug(startIndex, startIndex) = 1;
    //         X_aug.block(0,startIndex,3,1) = p + R*it->position;

    //         // Initialize new landmark covariance - TODO:speed up
    //         Eigen::MatrixXd F = Eigen::MatrixXd::Zero(state_.dimP()+3,state_.dimP());
    //         F.block(0,0,state_.dimP()-state_.dimTheta(),state_.dimP()-state_.dimTheta()) = Eigen::MatrixXd::Identity(state_.dimP()-state_.dimTheta(),state_.dimP()-state_.dimTheta()); // for old X
    //         F.block(state_.dimP()-state_.dimTheta(),6,3,3) = Eigen::Matrix3d::Identity(); // for new landmark
    //         F.block(state_.dimP()-state_.dimTheta()+3,state_.dimP()-state_.dimTheta(),state_.dimTheta(),state_.dimTheta()) = Eigen::MatrixXd::Identity(state_.dimTheta(),state_.dimTheta()); // for theta
    //         Eigen::MatrixXd G = Eigen::MatrixXd::Zero(F.rows(),3);
    //         G.block(G.rows()-state_.dimTheta()-3,0,3,3) = R;
    //         P_aug = (F*P_aug*F.transpose() + G*noise_params_.getLandmarkCov()*G.transpose()).eval();

    //         // Update state and covariance
    //         state_.setX(X_aug);
    //         state_.setP(P_aug);

    //         // Add to list of estimated landmarks
    //         estimated_landmarks_.insert(pair<int,int> (it->id, startIndex));
    //     }
    // }
    return;
}


// Create Observation from vector of landmark measurements
void InEKF::CorrectLandmarks(const vectorLandmarks& measured_landmarks) {

    Eigen::VectorXd Y;
    Eigen::VectorXd b;
    Eigen::MatrixXd H;
    Eigen::MatrixXd N;
    Eigen::MatrixXd PI;

    Eigen::Matrix3d R = state_.getRotation();
    vectorLandmarks new_landmarks;
    vector<int> used_landmark_ids;

    int lm_ind = 0;
    for (vectorLandmarksIterator it=measured_landmarks.begin(); it!=measured_landmarks.end(); ++it) {
        std::cout << "landmark " << lm_ind << std::endl;
        lm_ind ++;

        // Detect and skip if an ID is not unique (this would cause singularity issues in InEKF::Correct)
        if (find(used_landmark_ids.begin(), used_landmark_ids.end(), it->id) != used_landmark_ids.end()) {
            cout << "Duplicate landmark ID detected! Skipping measurement.\n";
            continue;
        } else { used_landmark_ids.push_back(it->id); }

        // See if we can find id in prior_landmarks or estimated_landmarks
        mapIntVector3dIterator it_prior = prior_landmarks_.find(it->id);
        map<int,int>::iterator it_estimated = estimated_landmarks_.find(it->id);
        if (it_prior!=prior_landmarks_.end()) {
            std::cout << "found in prior landmark set" << std::endl;
            // Found in prior landmark set
            int dimX = state_.dimX();
            int dimP = state_.dimP();
            int startIndex;

            // Fill out Y
            startIndex = Y.rows();
            Y.conservativeResize(startIndex+dimX, Eigen::NoChange);
            Y.segment(startIndex,dimX) = Eigen::VectorXd::Zero(dimX);
            Y.segment(startIndex,3) = it->position; // p_bl
            Y(startIndex+4) = 1;

            // Fill out b
            startIndex = b.rows();
            b.conservativeResize(startIndex+dimX, Eigen::NoChange);
            b.segment(startIndex,dimX) = Eigen::VectorXd::Zero(dimX);
            b.segment(startIndex,3) = it_prior->second; // p_wl
            b(startIndex+4) = 1;

            // Fill out H
            startIndex = H.rows();
            H.conservativeResize(startIndex+3, dimP);
            H.block(startIndex,0,3,dimP) = Eigen::MatrixXd::Zero(3,dimP);
            H.block(startIndex,0,3,3) = skew(it_prior->second); // skew(p_wl)
            H.block(startIndex,6,3,3) = -Eigen::Matrix3d::Identity(); // -I

            // Fill out N
            startIndex = N.rows();
            N.conservativeResize(startIndex+3, startIndex+3);
            N.block(startIndex,0,3,startIndex) = Eigen::MatrixXd::Zero(3,startIndex);
            N.block(0,startIndex,startIndex,3) = Eigen::MatrixXd::Zero(startIndex,3);
            N.block(startIndex,startIndex,3,3) = R * noise_params_.getLandmarkCov() * R.transpose();

            // Fill out PI
            startIndex = PI.rows();
            int startIndex2 = PI.cols();
            PI.conservativeResize(startIndex+3, startIndex2+dimX);
            PI.block(startIndex,0,3,startIndex2) = Eigen::MatrixXd::Zero(3,startIndex2);
            PI.block(0,startIndex2,startIndex,dimX) = Eigen::MatrixXd::Zero(startIndex,dimX);
            PI.block(startIndex,startIndex2,3,dimX) = Eigen::MatrixXd::Zero(3,dimX);
            PI.block(startIndex,startIndex2,3,3) = Eigen::Matrix3d::Identity();

        } else if (it_estimated!=estimated_landmarks_.end()) {;
            std::cout << "found in estimated landmark set" << std::endl;
            // Found in estimated landmark set
            int dimX = state_.dimX();
            int dimP = state_.dimP();
            int startIndex;

            std::cout << "Y" << std::endl;
            std::cout << "dimX: " << dimX << std::endl;
            std::cout << "dimP: " << dimP << std::endl;
            // Fill out Y
            startIndex = Y.rows();
            std::cout << "startIndex: " << startIndex << std::endl;
            std::cout << "it->second: " << it_estimated->second << std::endl;
            Y.conservativeResize(startIndex+dimX, Eigen::NoChange);
            Y.segment(startIndex,dimX) = Eigen::VectorXd::Zero(dimX);
            Y.segment(startIndex,3) = it->position; // p_bl
            Y(startIndex+4) = 1;
            Y(startIndex+it_estimated->second) = -1;

            std::cout << "b" << std::endl;
            // Fill out b
            startIndex = b.rows();
            b.conservativeResize(startIndex+dimX, Eigen::NoChange);
            b.segment(startIndex,dimX) = Eigen::VectorXd::Zero(dimX);
            b(startIndex+4) = 1;
            b(startIndex+it_estimated->second) = -1;

            std::cout << "H" << std::endl;
            // Fill out H
            startIndex = H.rows();
            H.conservativeResize(startIndex+3, dimP);
            H.block(startIndex,0,3,dimP) = Eigen::MatrixXd::Zero(3,dimP);
            H.block(startIndex,6,3,3) = -Eigen::Matrix3d::Identity(); // -I
            H.block(startIndex,3*it_estimated->second-6,3,3) = Eigen::Matrix3d::Identity(); // I

            // Fill out N
            startIndex = N.rows();
            N.conservativeResize(startIndex+3, startIndex+3);
            N.block(startIndex,0,3,startIndex) = Eigen::MatrixXd::Zero(3,startIndex);
            N.block(0,startIndex,startIndex,3) = Eigen::MatrixXd::Zero(startIndex,3);
            N.block(startIndex,startIndex,3,3) = R * noise_params_.getLandmarkCov() * R.transpose();

            // Fill out PI
            startIndex = PI.rows();
            int startIndex2 = PI.cols();
            PI.conservativeResize(startIndex+3, startIndex2+dimX);
            PI.block(startIndex,0,3,startIndex2) = Eigen::MatrixXd::Zero(3,startIndex2);
            PI.block(0,startIndex2,startIndex,dimX) = Eigen::MatrixXd::Zero(startIndex,dimX);
            PI.block(startIndex,startIndex2,3,dimX) = Eigen::MatrixXd::Zero(3,dimX);
            PI.block(startIndex,startIndex2,3,3) = Eigen::Matrix3d::Identity();


        } else {
            std::cout << "first time detected" << std::endl;
            // First time landmark as been detected (add to list for later state augmentation)
            new_landmarks.push_back(*it);
        }
    }

    std::cout << "Correcting state using stacked observations" << std::endl;
    // Correct state using stacked observation
    Observation obs(Y,b,H,N,PI);
    if (!obs.empty()) {
        this->Correct(obs);
    }

    // We don't need to augment state with landmarks because this is
    // handles by the MSCKF
    std::cout << "Updating estimated landmarks" << std::endl;
    int startIndex = state_.getX().rows();
    for (vectorLandmarksIterator it=new_landmarks.begin(); it!=new_landmarks.end(); ++it) {
        startIndex++;
        estimated_landmarks_.insert(pair<int,int> (it->id, startIndex));
    }

    // std::cout << "Augmenting state with new landmarks" << std::endl;
    // // Augment state with newly detected landmarks
    // if (new_landmarks.size() > 0) {
    //     Eigen::MatrixXd X_aug = state_.getX();
    //     Eigen::MatrixXd P_aug = state_.getP();
    //     Eigen::Vector3d p = state_.getPosition();
    //     for (vectorLandmarksIterator it=new_landmarks.begin(); it!=new_landmarks.end(); ++it) {
    //         // Initialize new landmark mean
    //         int startIndex = X_aug.rows();
    //         X_aug.conservativeResize(startIndex+1, startIndex+1);
    //         X_aug.block(startIndex,0,1,startIndex) = Eigen::MatrixXd::Zero(1,startIndex);
    //         X_aug.block(0,startIndex,startIndex,1) = Eigen::MatrixXd::Zero(startIndex,1);
    //         X_aug(startIndex, startIndex) = 1;
    //         X_aug.block(0,startIndex,3,1) = p + R*it->position;

    //         // Initialize new landmark covariance - TODO:speed up
    //         Eigen::MatrixXd F = Eigen::MatrixXd::Zero(state_.dimP()+3,state_.dimP());
    //         F.block(0,0,state_.dimP()-state_.dimTheta(),state_.dimP()-state_.dimTheta()) = Eigen::MatrixXd::Identity(state_.dimP()-state_.dimTheta(),state_.dimP()-state_.dimTheta()); // for old X
    //         F.block(state_.dimP()-state_.dimTheta(),6,3,3) = Eigen::Matrix3d::Identity(); // for new landmark
    //         F.block(state_.dimP()-state_.dimTheta()+3,state_.dimP()-state_.dimTheta(),state_.dimTheta(),state_.dimTheta()) = Eigen::MatrixXd::Identity(state_.dimTheta(),state_.dimTheta()); // for theta
    //         Eigen::MatrixXd G = Eigen::MatrixXd::Zero(F.rows(),3);
    //         G.block(G.rows()-state_.dimTheta()-3,0,3,3) = R;
    //         P_aug = (F*P_aug*F.transpose() + G*noise_params_.getLandmarkCov()*G.transpose()).eval();

    //         // Update state and covariance
    //         state_.setX(X_aug);
    //         state_.setP(P_aug);

    //         // Add to list of estimated landmarks
    //         estimated_landmarks_.insert(pair<int,int> (it->id, startIndex));
    //     }
    // }
    return;
}


void removeRowAndColumn(Eigen::MatrixXd& M, int index) {
    unsigned int dimX = M.cols();
    // cout << "Removing index: " << index<< endl;
    M.block(index,0,dimX-index-1,dimX) = M.bottomRows(dimX-index-1).eval();
    M.block(0,index,dimX,dimX-index-1) = M.rightCols(dimX-index-1).eval();
    M.conservativeResize(dimX-1,dimX-1);
}

} // end inekf namespace

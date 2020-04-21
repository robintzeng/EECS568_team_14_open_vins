
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <Eigen/Dense>
#include "RobotState.h"
#include "InEKF.h"

#define DT_MIN 1e-6
#define DT_MAX 1

using namespace std;
using namespace inekf;


int test_cam_pose(){
    cout << "Testing camera pose: " << endl;

    RobotState robot_state;
    assert(robot_state.getX().rows() == 5);
    assert(robot_state.getX().cols() == 5);


    Eigen::MatrixXd R = 5*Eigen::MatrixXd::Identity(3, 3);
    Eigen::Vector3d pose;
    pose << 1, 2, 3;

    CameraPose cam_pose(R, pose);

    robot_state.augmentState(cam_pose);

    cout << "X: \n" << robot_state.getX() << endl;
    cout << "P: \n" << robot_state.getP() << endl;

    assert(robot_state.getX().rows() == 5);
    assert(robot_state.getX().cols() == 9);

    assert(robot_state.getNumberCameras() == 1);

    robot_state.augmentState(cam_pose);
    assert(robot_state.getNumberCameras() == 2);
    assert(robot_state.getX().rows() == 5);
    assert(robot_state.getX().cols() == 13);

    // Test updating a camera pose with a new estimate
    CameraPose new_cam_pose(10*R, 0.5*pose);
    robot_state.updateCameraEstimate(1, new_cam_pose);
    cout << "X: \n" << robot_state.getX() << endl;
    cout << "P: \n" << robot_state.getP() << endl;

    return 0;

}


int main() {
    cout << "Testing" << endl;

    test_cam_pose();

    return 0;
}

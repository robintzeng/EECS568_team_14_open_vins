#include <gtest/gtest.h>

TEST(TestSuite, testCase1){
    int a = 1;
    int b = 1;

    EXPECT_EQ(a, b);
}


int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "tester");
    ros::NodeHandle nh;
    return RUN_ALL_TESTS();
}

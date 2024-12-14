#include <iostream>
#include <vector>
#include <getopt.h>
#include <string.h>
#include <time.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "inference.h"

using namespace std;
using namespace cv;

double fps;

std::chrono::time_point<std::chrono::system_clock> lastCollisionTimeHR, lastCollisionTimeRR;
int collisionCount = 0;
bool hasCollided = false;

// Vector untuk menyimpan semua peak
std::vector<int> peaks;

void calculateHR(float durationInSeconds) {
    if (durationInSeconds > 0) {
        float hr = 60.0 / durationInSeconds;
        std::cout << "Duration HR: " << durationInSeconds << " seconds" << std::endl;
        std::cout << "Average HR = " << hr << " BPM" << std::endl;
    }
}

void calculateRR(float durationInSeconds) {
    if (durationInSeconds > 0) {
        float rr = 60.0 / durationInSeconds;
        std::cout << "Duration RR: " << durationInSeconds << " seconds" << std::endl;
        std::cout << "Average RR = " << rr << " BPM" << std::endl;
    }
}

struct TimeCounter {
public:
    TimeCounter() : reset(true) {}

    float Count() {
        if (reset) {
            previous_time = std::chrono::system_clock::now();
            reset = false;
        }

        current_time = std::chrono::system_clock::now();
        elapsed_time = current_time - previous_time;

        return elapsed_time.count();
    }

    void Reset() {
        reset = true;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> current_time, previous_time;
    std::chrono::duration<float> elapsed_time;

    bool reset;
};

int getBinaryPeak(cv::Mat& roi) {
    int peak = 0;

    for (int i = 0; i < roi.rows; i++) {
        for (int j = 0; j < roi.cols; j++) {
            if (roi.at<uchar>(j, i) == 255) {
                peak = i;
                break;
            }
            if (peak != 0) {
                break;
            }
        }
    }
    return peak;
}

int main(int argc, char** argv) {
    TimeCounter time;
    VideoCapture cap("/home/eros/projek_HR/src/sample.mp4");

    if (!cap.isOpened()) {
        std::cout << "Cannot open video file" << std::endl;
        return -1;
    }

    bool runOnGPU = false;

    int m_size = 480;
    std::string model_path = "/home/eros/models/HR.onnx";

    Inference inf(model_path, cv::Size(m_size, m_size), "classes.txt", runOnGPU);

    int num_frames = 0;

    Detection peak_;

    while (true) {
        if (time.Count() >= 1) {
            num_frames = 0;
            time.Reset();
        }
        num_frames++;

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break; // Video selesai
        }

        int newWidth = 720;
        int newHeight = static_cast<int>((newWidth * frame.rows) / frame.cols);
        cv::resize(frame, frame, cv::Size(newWidth, newHeight));

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        int line = frame.cols / 2;
        cv::Mat frameClone = frame.clone();
        for (int i = 0; i < detections; ++i) {
            Detection detection = output[i];
            cv::Rect box = detection.box;

            cv::rectangle(frameClone, box, (detection.class_id == 1) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), 2);

            cv::line(frameClone, cv::Point2f(line, 0), cv::Point2f(line, frameClone.rows - 1), cv::Scalar(0, 255, 0), 1);
            if (output[i].class_id == 1) {
                peak_ = output[i];

                if (!hasCollided && peak_.box.x <= line && peak_.box.x + peak_.box.width >= line) {
                    auto currentCollisionTime = std::chrono::system_clock::now();

                    if (collisionCount == 0) {
                        lastCollisionTimeHR = currentCollisionTime;
                        lastCollisionTimeRR = currentCollisionTime;
                    }
                    else {
                        std::chrono::duration<float> durationHR = currentCollisionTime - lastCollisionTimeHR;
                        float durationInSecondsHR = durationHR.count();
                        calculateHR(durationInSecondsHR);
                        lastCollisionTimeHR = currentCollisionTime;

                        if (collisionCount % 2 == 0) {
                            std::chrono::duration<float> durationRR = currentCollisionTime - lastCollisionTimeRR;
                            float durationInSecondsRR = durationRR.count();
                            calculateRR(durationInSecondsRR);
                            lastCollisionTimeRR = currentCollisionTime;
                        }
                    }

                    hasCollided = true;
                    collisionCount++;

                    cv::Mat roi = frame(peak_.box);
                    cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);

                    cv::Mat roiBinary;
                    cv::threshold(roi, roiBinary, 200, 255, cv::THRESH_BINARY);

                    cv::Mat roiInverted;
                    cv::bitwise_not(roiBinary, roiInverted);

                    if (!roi.empty()) {
                        cv::imshow("Invert", roiInverted);
                    }

                    // Perhitungan peak
                    int binaryPeak = getBinaryPeak(roiInverted);
                    int framePeak = peak_.box.y + (binaryPeak / 2);

                    // Simpan peak ke vector
                    peaks.push_back(framePeak);

                    cv::line(frameClone, cv::Point2f(0, framePeak), cv::Point2f(frameClone.cols, framePeak), cv::Scalar(255, 0, 255), 1);
                }
                else if (peak_.box.x + peak_.box.width <= line) {
                    hasCollided = false;
                }
            }
        }

        cv::imshow("Inference", frameClone);
        if (waitKey(1) == 27) break;
    }

    // Setelah video selesai, hitung rata-rata peak
    if (!peaks.empty()) {
        int totalPeak = std::accumulate(peaks.begin(), peaks.end(), 0);
        float averagePeak = static_cast<float>(totalPeak) / peaks.size();

        std::cout << "Total Peaks Detected: " << peaks.size() << std::endl;
        std::cout << "Average Peak Position: " << averagePeak << std::endl;
    }

    return 0;
}

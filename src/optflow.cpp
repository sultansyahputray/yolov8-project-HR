#include <iostream>
#include <vector>
#include <getopt.h>
#include <string.h>
#include <time.h>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "yaml-cpp/yaml.h"
#include "inference.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>

using namespace std;
using namespace cv;

double fps;
cv::Mat rr_mat;
int HR_, RR_;

std::vector<int> respiration_rates;

std::chrono::time_point<std::chrono::system_clock> lastCollisionTimeHR, lastCollisionTimeRR;
int collisionCount = 0;

bool isCounting(int threshold, int a, int b) {
    return a <= threshold && b >= threshold; 
}

void calculateHR(float durationInSeconds) {
    if (durationInSeconds > 0) {
        HR_ = 60.0 / durationInSeconds;  
        std::cout << "Average HR = " << HR_ << std::endl;  
    }
}

void calculateRR(float durationInSeconds) {
    if (durationInSeconds > 0) {
        RR_ = 60.0 / durationInSeconds;  
        std::cout << "Average RR = " << RR_ << std::endl;
    }
}

void updateRespirationRates(std::vector<int>& respiration_rates, int newRate, bool& isNewData) {
    // Jika ada data baru, tambahkan ke vector
    if (isNewData) {
        respiration_rates.push_back(newRate);

        // Hapus elemen pertama jika vector lebih dari ukuran tertentu
        if (respiration_rates.size() > 100) {
            respiration_rates.erase(respiration_rates.begin());
        }
    }

    // Tandai jika data baru diterima
    isNewData = false;
}

void drawRR(std::vector<int>& dots, cv::Mat& canvas, bool hasNewData) {
    auto p = std::minmax_element(dots.begin(), dots.end());

    int x_length = canvas.cols / (dots.size() + 1);
    int y_length = canvas.rows / (*p.second - *p.first + 2);

    // Jika tidak ada data baru, hanya geser titik-titik yang ada
    if (!hasNewData) {
        std::rotate(dots.begin(), dots.begin() + 1, dots.end()); 
    }

    for(int i = 0; i < dots.size(); i++){
        int x_pos = x_length * (i + 1);
        int y_pos = canvas.rows - (y_length * dots[i]);

        cv::circle(canvas, cv::Point(x_pos, y_pos), 1, cv::Scalar(0, 255, 0), -1);

        if(i < dots.size() - 1){
            cv::Point next = cv::Point(x_pos + x_length, canvas.rows - (y_length * dots[i + 1]));
            cv::line(canvas, cv::Point(x_pos, y_pos), next, cv::Scalar(255, 0, 0), 1);
        }
    }
}

// Fungsi untuk menangkap gambar dari window tertentu berdasarkan window ID
cv::Mat captureScreen(Window windowID) {
    Display* display = XOpenDisplay(NULL);
    if (display == NULL) {
        std::cerr << "Gagal membuka display!" << std::endl;
        exit(1);
    }

    // Ambil ukuran jendela yang ingin di-capture
    XWindowAttributes windowAttributes;
    XGetWindowAttributes(display, windowID, &windowAttributes);
    int width = windowAttributes.width;
    int height = windowAttributes.height;

    // Tangkap gambar dari window tertentu
    XImage* image = XGetImage(display, windowID, 0, 0, width, height, AllPlanes, ZPixmap);
    if (!image) {
        std::cerr << "Gagal menangkap jendela!" << std::endl;
        exit(1);
    }

    // Salin data dari XImage ke OpenCV Mat
    cv::Mat screenMat(height, width, CV_8UC4);
    memcpy(screenMat.data, image->data, height * width * 4);

    // Hapus XImage setelah digunakan
    XDestroyImage(image);
    XCloseDisplay(display);

    // Konversi dari format BGRA ke BGR (OpenCV menggunakan BGR)
    cv::Mat bgrMat;
    cv::cvtColor(screenMat, bgrMat, cv::COLOR_BGRA2BGR);

    return bgrMat;
}

struct TimeCounter {
    public:
    TimeCounter() : reset(true) {}
  
    float Count() {
        if (reset) {
            previous_time   = std::chrono::system_clock::now();
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

std::chrono::duration<float> maxInactiveDuration = std::chrono::seconds(5);  // Batas waktu 10 detik tanpa update

void checkAndResetRespirationRates() {
    // Menghitung selisih waktu antara waktu tabrakan terakhir HR dan saat ini
    auto currentCollisionTime = std::chrono::system_clock::now();

    std::chrono::duration<float> durationSinceHR = currentCollisionTime - lastCollisionTimeHR;
    std::chrono::duration<float> durationSinceRR = currentCollisionTime - lastCollisionTimeRR;

    // reset respiration_rates
    if (durationSinceHR > maxInactiveDuration && durationSinceRR > maxInactiveDuration) {
        respiration_rates.clear();  // Menghapus semua data dalam respiration_rates
        std::cout << "Respiration rates reset due to inactivity over 10 seconds." << std::endl;
    }
}

int main(int argc, char **argv) {
    TimeCounter time;
    bool isNewData = false;  // Menandakan apakah ada data baru

    // init RR
    respiration_rates.push_back(0);  

    bool runOnGPU = false;
    std::string model_path = "/home/eros/Downloads/yolov8-project-HR/src/80.onnx";
    int m_size = 480;
    Inference inf(model_path, cv::Size(m_size, m_size), "classes.txt", runOnGPU);

    int num_frames = 0;
    clock_t start, end;
    double ms, fpsLive;
    int threshold = 200;
    bool toggle = false;
    int hitung = 0;

    cv::VideoCapture cap;
    // Ambil window ID dari argumen input
    Window windowID;
    if (argc < 2) {
        cap = cv::VideoCapture("/home/eros/project_HR/sample/80sample.mp4");
    } else {
        windowID = (Window)strtol(argv[1], nullptr, 16);
    }

    std::chrono::time_point<std::chrono::system_clock> lastUpdateTime = std::chrono::system_clock::now();

    while(true) {   
        if(time.Count() >= 1) {
            num_frames = 0;
            time.Reset();
        }
        num_frames++;
        start = clock(); 
        
        fps = 30;  // Tentukan FPS atau bisa ditangkap dari `captureScreen`

        // Tangkap layar dari window tertentu
        cv::Mat frame;
        if (argc < 2) {
            cap >> frame;  // Capture frame from the video.
        } else {
            frame = captureScreen(windowID);  // Capture screen based on windowID.
        }

        std::vector<Detection> output = inf.runInference(frame);

        std::string peak = "jumlah peak : " + std::to_string(hitung); 
        std::string displayHR = "HR : " + std::to_string(HR_); 
        std::string displayRR = "RR : " + std::to_string(RR_); 

        cv::putText(frame, peak, Point(20, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);
        cv::putText(frame, displayHR, Point(400, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);
        cv::putText(frame, displayRR, Point(600, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);

        int detections = output.size();

        cv::Mat frameClone = frame.clone();
        cv::line(frameClone, cv::Point2f(threshold, 0), cv::Point2f(threshold, frameClone.rows - 1), cv::Scalar(0, 255, 0), 1);
      
        for (int i = 0; i < detections; ++i) {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;
            detection.isCount = (detection.class_id == 1) ? isCounting(threshold, detection.box.x, detection.box.x + detection.box.width) : false; 

            if(detection.box.x <= threshold && detection.box.x + detection.box.width > threshold) {
                if(detection.class_id == 1 && toggle) {
                    toggle = false;
                    hitung++;

                    auto currentCollisionTime = std::chrono::system_clock::now();

                    if (collisionCount == 0) {
                        lastCollisionTimeHR = currentCollisionTime;
                        lastCollisionTimeRR = currentCollisionTime;
                    } else {
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

                    collisionCount++;
                }

                if(detection.class_id == 0) {
                    toggle = true;
                }
            }

            if(detection.class_id == 0) {
                cv::rectangle(frameClone, box, cv::Scalar(0, 0, 255), 2);
            } else {
                cv::rectangle(frameClone, box, (detection.isCount) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0), 2);
            }
        }

        end = clock();
        float scale = 0.5;
        cv::resize(frameClone, frameClone, cv::Size(frameClone.cols*scale, frameClone.rows*scale));

        rr_mat = cv::Mat::zeros(cv::Size(frameClone.cols, frameClone.rows / 2), frameClone.type());

        // Update respiration rates every 0.1 seconds
        auto currentTime = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed = currentTime - lastUpdateTime;

        if (elapsed.count() >= 0.1) {
            // Set isNewData flag to true if RR data has changed
            isNewData = true;
            updateRespirationRates(respiration_rates, RR_, isNewData);
            lastUpdateTime = currentTime;
        }

        checkAndResetRespirationRates();

        if (respiration_rates.size() != 0) {
            drawRR(respiration_rates, rr_mat, isNewData);
        }

        double sc = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
        fpsLive = double(num_frames) / double(sc);

        cv::Mat combine;
        cv::vconcat(frameClone, rr_mat, combine);

        cv::imshow("Inference", combine);

        if(waitKey(1) == 27) break;
    }

    return 0;
}

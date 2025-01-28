#include <iostream>
#include <vector>
#include <getopt.h>
#include <string.h>
#include <time.h>
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

int HR_;

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

int main(int argc, char **argv) {
    TimeCounter time;

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

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <window_id>" << std::endl;
        return 1;
    }

    // Ambil window ID dari argumen input
    Window windowID = (Window)strtol(argv[1], nullptr, 16);

    while(true) {   
        if(time.Count() >= 1) {
            num_frames = 0;
            time.Reset();
        }
        num_frames++;
        start = clock(); 
        
        fps = 30;  // Tentukan FPS atau bisa ditangkap dari captureScreen

        // Tangkap layar dari window tertentu
        cv::Mat frame = captureScreen(windowID);  

        std::vector<Detection> output = inf.runInference(frame);

        std::string peak = "jumlah peak : " + std::to_string(hitung); 
        std::string displayHR = "HR : " + std::to_string(HR_);  

        cv::putText(frame, peak, Point(20, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);
        cv::putText(frame, displayHR, Point(400, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);

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
            std::cout << hitung << std::endl;
        }

        end = clock();
        float scale = 0.5;
        cv::resize(frameClone, frameClone, cv::Size(frameClone.cols*scale, frameClone.rows*scale));

        double sc = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
        fpsLive = double(num_frames) / double(sc);

        cv::imshow("Inference", frameClone);

        if(waitKey(1) == 27) break;
    }

    return 0;
}
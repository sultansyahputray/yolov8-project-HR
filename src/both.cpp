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

int HR_, RR_; // HR dan RR untuk masing-masing jendela

std::chrono::time_point<std::chrono::system_clock> lastCollisionTimeHR, lastCollisionTimeRR;
int collisionCount1 = 0, collisionCount2 = 0;

bool isCounting(int threshold, int a, int b) {
    return a <= threshold && b >= threshold; 
}

void calculateHR(float durationInSeconds, int &HR) {
    if (durationInSeconds > 0) {
        HR = 60.0 / durationInSeconds;  
        std::cout << "Average HR = " << HR << std::endl;  
    }
}

void calculateRR(float durationInSeconds, int &RR) {
    if (durationInSeconds > 0) {
        RR = 60.0 / durationInSeconds;  
        std::cout << "Average RR = " << RR << std::endl;
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

    // Model yang berbeda untuk jendela 1 dan jendela 2
    std::string model_path1 = "/home/eros/Downloads/yolov8-project-HR/src/80.onnx";  // Model untuk window 1
    std::string model_path2 = "/home/eros/Downloads/yolov8-project-HR/src/HR.onnx";  // Model untuk window 2
    int m_size = 480;
    
    // Objek inference yang berbeda untuk masing-masing jendela
    Inference inf1(model_path1, cv::Size(m_size, m_size), "classes.txt", runOnGPU);  // Model untuk window 1
    Inference inf2(model_path2, cv::Size(m_size, m_size), "classes.txt", runOnGPU);  // Model untuk window 2

    int num_frames = 0;
    clock_t start, end;
    double ms, fpsLive;
    int threshold = 200;
    bool toggle1 = false, toggle2 = false;
    int hitung1 = 0, hitung2 = 0;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <window_id_1> <window_id_2>" << std::endl;
        return 1;
    }

    // Ambil window ID dari argumen input
    Window windowID1 = (Window)strtol(argv[1], nullptr, 16);
    Window windowID2 = (Window)strtol(argv[2], nullptr, 16);

    while (true) {   
        if (time.Count() >= 1) {
            num_frames = 0;
            time.Reset();
        }
        num_frames++;
        start = clock(); 
        
        fps = 30;  // Tentukan FPS atau bisa ditangkap dari `captureScreen`

        // Tangkap layar dari kedua window
        cv::Mat frame1 = captureScreen(windowID1);  
        cv::Mat frame2 = captureScreen(windowID2);  

        // Proses untuk window 1 (menggunakan model1)
        std::vector<Detection> output1 = inf1.runInference(frame1);
        std::string peak1 = "jumlah peak : " + std::to_string(hitung1); 
        std::string displayHR = "HR : " + std::to_string(HR_); 

        cv::putText(frame1, peak1, Point(20, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);
        cv::putText(frame1, displayHR, Point(400, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);

        int detections1 = output1.size();
        cv::Mat frameClone1 = frame1.clone();
        cv::line(frameClone1, cv::Point2f(threshold, 0), cv::Point2f(threshold, frameClone1.rows - 1), cv::Scalar(0, 255, 0), 1);

        for (int i = 0; i < detections1; ++i) {
            Detection detection = output1[i];

            cv::Rect box = detection.box;
            detection.isCount = (detection.class_id == 1) ? isCounting(threshold, detection.box.x, detection.box.x + detection.box.width) : false; 

            if (detection.box.x <= threshold && detection.box.x + detection.box.width > threshold) {
                if (detection.class_id == 1 && toggle1) {
                    toggle1 = false;
                    hitung1++;

                    auto currentCollisionTime = std::chrono::system_clock::now();
                    std::chrono::duration<float> durationHR = currentCollisionTime - lastCollisionTimeHR;
                    float durationInSecondsHR = durationHR.count();
                    calculateHR(durationInSecondsHR, HR_);
                    lastCollisionTimeHR = currentCollisionTime; 

                    collisionCount1++;
                }

                if (detection.class_id == 0) {
                    toggle1 = true;
                }
            }

            cv::rectangle(frameClone1, box, (detection.isCount) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0), 2);
        }

        // Proses untuk window 2 (menggunakan model2)
        std::vector<Detection> output2 = inf2.runInference(frame2);
        std::string peak2 = "jumlah peak : " + std::to_string(hitung2); 
        std::string displayRR = "RR : " + std::to_string(RR_); 

        cv::putText(frame2, peak2, Point(20, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);
        cv::putText(frame2, displayRR, Point(400, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(170, 0, 170), 2, 0);

        int detections2 = output2.size();
        cv::Mat frameClone2 = frame2.clone();
        cv::line(frameClone2, cv::Point2f(threshold, 0), cv::Point2f(threshold, frameClone2.rows - 1), cv::Scalar(0, 255, 0), 1);

        for (int i = 0; i < detections2; ++i) {
            Detection detection = output2[i];

            cv::Rect box = detection.box;
            detection.isCount = (detection.class_id == 1) ? isCounting(threshold, detection.box.x, detection.box.x + detection.box.width) : false; 

            if (detection.box.x <= threshold && detection.box.x + detection.box.width > threshold) {
                if (detection.class_id == 1 && toggle2) {
                    toggle2 = false;
                    hitung2++;

                    auto currentCollisionTime = std::chrono::system_clock::now();
                    std::chrono::duration<float> durationRR = currentCollisionTime - lastCollisionTimeRR;
                    float durationInSecondsRR = durationRR.count();
                    calculateRR(durationInSecondsRR, RR_);
                    lastCollisionTimeRR = currentCollisionTime; 

                    collisionCount2++;
                }

                if (detection.class_id == 0) {
                    toggle2 = true;
                }
            }

            cv::rectangle(frameClone2, box, (detection.isCount) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0), 2);
        }

        end = clock();
        float scale = 0.5;
        cv::resize(frameClone1, frameClone1, cv::Size(frameClone1.cols * scale, frameClone1.rows * scale));
        cv::resize(frameClone2, frameClone2, cv::Size(frameClone2.cols * scale, frameClone2.rows * scale));

        // Gabungkan frame 1 dan frame 2 secara vertikal
        cv::Mat combinedFrame;
        cv::vconcat(frameClone1, frameClone2, combinedFrame);

        // Menampilkan frame gabungan
        cv::imshow("Inference", combinedFrame);

        // Hitung FPS
        double sc = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
        fpsLive = double(num_frames) / double(sc);

        if (waitKey(1) == 27) break;
    }

    return 0;
}

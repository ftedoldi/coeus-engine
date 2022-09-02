#ifndef __STATUS_BAR_H__
#define __STATUS_BAR_H__

#include <vector>
#include <string>

namespace EditorLayer {
    enum StatusBarTextColor {
        WHITE,
        RED,
        YELLOW,
        GREEN
    };

    struct StatusBarInfo {
        std::string statusText;
        StatusBarTextColor statusTextColor;
    };
    
    class StatusBar {
        private:
            std::vector<StatusBarInfo> statusQueue;

        public:
            StatusBarInfo errorStatus;
            
            StatusBar();

            bool isEmpty();

            void addStatus(const std::string& text, StatusBarTextColor textColor = StatusBarTextColor::WHITE);

            StatusBarInfo popStatus();
            StatusBarInfo popStatus(const short& i);

            StatusBarInfo getLastStatus();

            void draw();
    };
}

#endif __STATUS_BAR_H__
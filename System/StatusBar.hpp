#ifndef __STATUS_BAR_H__
#define __STATUS_BAR_H__

#include <vector>
#include <string>

namespace System {
    enum TextColor {
        WHITE,
        RED,
        YELLOW,
        GREEN
    };

    struct CurrentStatus {
        std::string statusText;
        TextColor statusTextColor;
    };
    
    class StatusBar {
        private:
            std::vector<CurrentStatus> statusQueue;

        public:
            CurrentStatus errorStatus;
            
            StatusBar();

            bool isEmpty();

            void addStatus(const std::string& text, TextColor textColor = TextColor::WHITE);

            CurrentStatus popStatus();
            CurrentStatus popStatus(const short& i);

            CurrentStatus getLastStatus();
    };
}

#endif __STATUS_BAR_H__
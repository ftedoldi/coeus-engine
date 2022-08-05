#include "../StatusBar.hpp"

namespace System {

    StatusBar::StatusBar()
    {
        errorStatus.statusText = " ";
        errorStatus.statusTextColor = TextColor::RED;
    }

    bool StatusBar::isEmpty()
    {
        return this->statusQueue.size() == 0;
    }

    void StatusBar::addStatus(const std::string& text, TextColor textColor)
    {
        CurrentStatus status;

        status.statusText = text;
        status.statusTextColor = textColor;

        if (getLastStatus().statusText != text)
            this->statusQueue.push_back(status);
    }

    CurrentStatus StatusBar::popStatus()
    {
        CurrentStatus lastStatus = this->statusQueue[this->statusQueue.size() - 1];
        this->statusQueue.pop_back();

        return lastStatus;
    }

    CurrentStatus StatusBar::popStatus(const short& i)
    {
        CurrentStatus poppedStatus = this->statusQueue[this->statusQueue.size() - 1];
        this->statusQueue.erase(this->statusQueue.begin() + i);

        return poppedStatus;
    }

    CurrentStatus StatusBar::getLastStatus()
    {
        if (this->statusQueue.size() > 0)
            return this->statusQueue[this->statusQueue.size() - 1];
        
        return errorStatus;
    }

}
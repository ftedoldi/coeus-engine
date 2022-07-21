#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <iostream>
#include <string>
#include <filesystem>

namespace System {
    class Folder {
        public:
            static std::string getApplicationAbsolutePath() {
                return std::filesystem::current_path().string();
            }

            static std::filesystem::path getFolderPath(std::string folderName) {
                for (auto& directory : std::filesystem::directory_iterator(Folder::getApplicationAbsolutePath())) {
                    auto& path = directory.path();
                    if (path.filename().string() == folderName)
                        return path;
                }

                return nullptr;
            }
    };
}

#endif // __SYSTEM_H__
#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <iostream>
#include <string>
#include <filesystem>

namespace System {
    class Folder {
        public:
            static std::filesystem::path assetDirectory;
            static std::filesystem::path currentDirectory;

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

            static int countNestedFolders(std::filesystem::path sourceFolder)
            {
                auto counter = 0;

                for (auto& directory : std::filesystem::directory_iterator(sourceFolder)) {
                    if (directory.is_directory())
                        ++counter;
                }

                return counter;
            }
    };
}

#endif // __SYSTEM_H__
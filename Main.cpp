#include <Window.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <Component.hpp>
#include <coeus.hpp>
#include <Shader.hpp>
#include <EditorCamera.hpp>
#include <Model.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Texture2D.hpp>
#include <SceneObject.hpp>
#include <SceneManager.hpp>
#include <Time.hpp>
#include "Test/DummyComponent.hpp"
#include <PointLight.hpp>
#include <DirectionalLight.hpp>
#include <Cubemap.hpp>
#include <SpotLight.hpp>
#include <AreaLight.hpp>
#include <RigidPlane.hpp>
#include <BasicCubemap.hpp>

#include <Serializer/Serializer.hpp>

#include <SerializableClass.hpp>

#include <rttr/registration>

#include <iostream>
#include <vector>
#include <thread>
#include <any>
#include <unordered_map>

#include "Scripts/CameraMovement.hpp"

#include <Python.h>

#include <FirstClass.cuh>

int main()
{
    System::Serialize::Serializer serializer = System::Serialize::Serializer();

    // Where all the starts are runned
    System::Window* window = new System::Window("Coeus Engine");

    std::ifstream stream("./Assets/projectSettings.meta");
    std::stringstream strStream;
    strStream << stream.rdbuf();

    YAML::Node data = YAML::Load(strStream.str());
    std::string scenePathToLoad;

    if (!data["Default Scene"] || !std::filesystem::exists(data["Default Scene"].as<std::string>()))
        scenePathToLoad = "./Assets/Scenes/EmptyScene.coeus";
    else
        scenePathToLoad = data["Default Scene"].as<std::string>();

    // PBR cubemap creation
    stbi_set_flip_vertically_on_load(true);
    Odysseus::Cubemap* mainCubemap = new Odysseus::Cubemap();
    Odysseus::Cubemap::currentCubemap = mainCubemap;
    stbi_set_flip_vertically_on_load(false);

    //Odysseus::BasicCubemap* phongCubemap = new Odysseus::BasicCubemap();
    
    serializer.deserialize(scenePathToLoad);

    //Setup everything before initializeScene call
    Odysseus::SceneManager::initializeActiveScene();

    // render loop
    // -----------
    while (!window->shouldWindowClose())
    {
        // render
        // ------
        window->clear();
        
        Odysseus::SceneManager::drawActiveScene();
        //phongCubemap->update();
        mainCubemap->update();

        window->update();
    }

    delete window;
    return 0;
}
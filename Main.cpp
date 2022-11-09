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

    stbi_set_flip_vertically_on_load(true);
    Odysseus::Cubemap* mainCubemap = new Odysseus::Cubemap();
    Odysseus::Cubemap::currentCubemap = mainCubemap;
    stbi_set_flip_vertically_on_load(false);
    
    serializer.deserialize(scenePathToLoad);

    // call();
    // AddVectors();
    // Athena::Vector3 test = Athena::Vector3(1, 1, 1);
    // Athena::Vector3 test1 = Athena::Vector3(1, 2, 3);
    // (test + test1).print();
    // Athena::Vector3 r = Athena::Vector3();
    // float* t = test.asScalarVector();
    // float* t1 = test1.asScalarVector();
    // float* result = r.asScalarVector();
    // auto res = AddVector3(t, t1, result);
    // r = Athena::Vector3(result);
    // r.print();


    //stbi_set_flip_vertically_on_load(true);

    //auto m1 = Odysseus::SceneManager::activeScene->objectsInScene[4]->getComponent<Odysseus::Mesh>();
    //auto m2 = myModel.objectsCreated[1]->getComponent<Odysseus::Mesh>();

    //m1->vertices = m2->vertices;
    //m1->indices = m2->indices;
    //m1->shader = m2->shader;

    // Odysseus::Model myModel(
    //                        "Assets\\Models\\PBRsphere\\PBRsphere.obj",
    //                        mainCubemap->PBRshader,
    //                        true
    //                    );

    Odysseus::Shader* myShader = new Odysseus::Shader("Shader\\phongShader.vert", "Shader\\toonShader.frag");
    Odysseus::Model model("Assets\\Models\\PBRsphere\\PBRsphere.obj", myShader, true, "obj");

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
        mainCubemap->update();

        window->update();
    }

    delete window;
    return 0;
}
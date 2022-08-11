#include <Window.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <coeus.hpp>
#include <Shader.hpp>
#include <camera.hpp>
#include <Model.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Texture2D.hpp>
#include <SceneObject.hpp>
#include <SceneGraph.hpp>
#include <Component.hpp>
#include <Time.hpp>
#include "Test/DummyComponent.hpp"
#include <PointLight.hpp>
#include <DirectionalLight.hpp>
#include <Cubemap.hpp>
#include <SpotLight.hpp>
#include <AreaLight.hpp>

#include <Serializer/Serializer.hpp>

#include <iostream>
#include <vector>

#include "Scripts/CameraMovement.hpp"

//#include <Test.cuh>

int main()
{
    System::Serialize::Serializer serializer = System::Serialize::Serializer();

    // Where all the starts are runned
    System::Window* window = new System::Window("myWindow");

    //Dummy component test
    System::Component *c = new DummyComponent();
    Odysseus::SceneObject *obj = new Odysseus::SceneObject();
    obj->transform->name = "Object";
    obj->addComponent<DummyComponent>();

    //Camera setup
    Odysseus::SceneObject *cam = new Odysseus::SceneObject();
    cam->transform->position = Athena::Vector3(0, 0, 20);
    cam->transform->name = "Camera";
    auto mainCamera = cam->addComponent<Odysseus::Camera>();
    Odysseus::Camera::main = mainCamera;

    //Camera movement setup
    auto movement = cam->addComponent<CameraMovement>();
    movement->camera = mainCamera;
    cam->getComponent<CameraMovement>()->camera = cam->getComponent<Odysseus::Camera>();

    //-------------------------------------------------------
    //Shaders setup

    Odysseus::Shader* modelShader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
    Odysseus::Shader* PBRshader = new Odysseus::Shader(".\\Shader\\PBRshader.vert", ".\\Shader\\PBRshader.frag");

    //-------------------------------------------------------
    //Models setup

    Odysseus::Model myModel("Assets/Models/PBRsphere/PBRsphere.obj", modelShader, false);
    Odysseus::Model myModel1("Assets/Models/matAndTex/matAndTex.obj", modelShader, false);
    // Odysseus::Model myModel1("Assets/Models/testFiles/posCubesColored.gltf", modelShader, false);
    // myModel.setIfPBR(true);
    // myModel1.setIfPBR(true);

    //-------------------------------------------------------
    //Light setup

    //Point light
    Odysseus::SceneObject *light = new Odysseus::SceneObject();
    light->transform->name = "PointLight";
    auto pLight = light->addComponent<Odysseus::PointLight>();
    
    pLight->setPosition(Athena::Vector3(0.0f, 2.0f, 0.0f));
    pLight->setShader(modelShader);
    pLight->setAmbient(Athena::Vector3(0.2f, 0.2f, 0.2f));
    pLight->setDiffuse(Athena::Vector3(0.8f, 0.8f, 0.8f));
    pLight->setSpecular(Athena::Vector3(0.3f, 0.3f, 0.3f));
    pLight->setConstant(1.0f);
    pLight->setLinear(0.09f);
    pLight->setQuadratic(0.032f);

    //Directional light
    Odysseus::SceneObject *dirLight = new Odysseus::SceneObject();
    dirLight->transform->name = "DirectionaLight";
    auto dLight = dirLight->addComponent<Odysseus::DirectionalLight>();

    dLight->setShader(modelShader);
    dLight->setAmbient(Athena::Vector3(0.2f, 0.2f, 0.2f));
    dLight->setDiffuse(Athena::Vector3(0.5f, 0.5f, 0.5f));
    dLight->setSpecular(Athena::Vector3(0.5f, 0.5f, 0.5f));
    dLight->setDirection(Athena::Vector3(0.0f, -1.0f, 0.0f));
    
    //Spot light
    Odysseus::SceneObject *spotLight = new Odysseus::SceneObject();
    spotLight->transform->name = "spotLight";
    auto sLight = spotLight->addComponent<Odysseus::SpotLight>();
    
    sLight->setPosition(Athena::Vector3(0.0f, 2.0f, 0.0f));
    sLight->setDirection(Athena::Vector3(0.0f, -1.0f, 0.0f));
    sLight->setShader(modelShader);
    sLight->setAmbient(Athena::Vector3(0.2f, 0.2f, 0.2f));
    sLight->setDiffuse(Athena::Vector3(0.8f, 0.8f, 0.8f));
    sLight->setSpecular(Athena::Vector3(1.0f, 1.0f, 1.0f));
    sLight->setCutOff(41.0f);
    sLight->setSpotExponent(19.0f);

    stbi_set_flip_vertically_on_load(true);
    
    //HDR map setup
    // Odysseus::Cubemap* HDRImap = new Odysseus::Cubemap();
    // HDRImap->setPBRshader(PBRshader);

    //Setup everything before initializeScene call
    Odysseus::SceneGraph::initializeScene();
    Odysseus::SceneGraph::name = "Base Scene";

    serializer.serialize("Assets/Scenes/Test.coeus");

    // render loop
    // -----------
    while (!window->shouldWindowClose())
    {
        // render
        // ------
        window->clear();
        
        Odysseus::SceneGraph::drawScene();

        // HDRImap->update();
        
        window->update();
    }

    delete window;
    return 0;
}
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

#include <iostream>
#include <vector>

#include "Scripts/CameraMovement.hpp"

//#include <Test.cuh>

int main()
{
    // Where all the starts are runned
    System::Window* window = new System::Window("myWindow");

    System::Component *c = new DummyComponent();
    Odysseus::SceneObject *obj = new Odysseus::SceneObject();
    obj->transform->name = "Object";
    obj->addComponent<DummyComponent>();

    Odysseus::SceneObject *cam = new Odysseus::SceneObject();
    cam->transform->position = Athena::Vector3(0, 0, 20);
    cam->transform->name = "Camera";
    auto mainCamera = cam->addComponent<Odysseus::Camera>();
    Odysseus::Camera::main = mainCamera;
    Odysseus::SceneObject *light = new Odysseus::SceneObject();
    light->transform->name = "PointLight";
    auto pLight = light->addComponent<Odysseus::PointLight>();

    /*Odysseus::SceneObject *dirLight = new Odysseus::SceneObject();
    dirLight->transform->name = "DirectionaLight";
    auto dLight = dirLight->addComponent<Odysseus::DirectionalLight>();*/
    // auto movement = cam->addComponent<CameraMovement>();
    // movement->camera = mainCamera;
    // cam->getComponent<CameraMovement>()->camera = cam->getComponent<Odysseus::Camera>();
    // cam->addComponent<Odysseus::Camera>();

    stbi_set_flip_vertically_on_load(true);

    // Create the shader
    Odysseus::Shader* modelShader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
    // Odysseus::Shader* PBRshader = new Odysseus::Shader(".\\Shader\\PBRshader.vert", ".\\Shader\\PBRshader.frag");
    Odysseus::Shader* lightShader = new Odysseus::Shader(".\\Shader\\lightShader.vert", ".\\Shader\\lightShader.frag");

    // Odysseus::Model myModel("Assets/Models/PBRcube/PBRcube.obj", PBRshader);
    Odysseus::Model myModel("Assets/Models/matAndTex/matAndTex.obj", modelShader);
    // Odysseus::Model myModel1("Assets/Models/PBRsphere/PBRsphere.obj", PBRshader);
    Odysseus::Model lightModel("Assets/Models/cubeCentered/cubeCentered.obj", lightShader);

    for(int i = 0; i < myModel.objectsCreated.size(); i++)
        myModel.objectsCreated[i]->transform->name = "MyModel" + std::to_string(i);

    for(int i = 0; i < lightModel.objectsCreated.size(); i++)
        lightModel.objectsCreated[i]->transform->name = "Light" + std::to_string(i);

    pLight->setPosition(Athena::Vector3(0.0f, 2.0f, 0.0f));
    pLight->setShader(lightShader);
    pLight->setAmbient(Athena::Vector3(0.2f, 0.2f, 0.2f));
    pLight->setDiffuse(Athena::Vector3(0.8f, 0.8f, 0.8f));
    pLight->setSpecular(Athena::Vector3(1.0f, 1.0f, 1.0f));
    pLight->setConstant(1.0f);
    pLight->setLinear(0.09f);
    pLight->setQuadratic(0.032f);

    /*dLight->setShader(modelShader);
    dLight->setAmbient(Athena::Vector3(0.2f, 0.2f, 0.2f));
    dLight->setDiffuse(Athena::Vector3(0.5f, 0.5f, 0.5f));
    dLight->setSpecular(Athena::Vector3(0.5f, 0.5f, 0.5f));
    dLight->setDirection(Athena::Vector3(0.0f, -1.0f, 0.0f));*/
    // Odysseus::Cubemap HDRImap;
    // HDRImap.setPBRshader(PBRshader);

    Odysseus::SceneGraph::initializeScene();

    // render loop
    // -----------
    while (!window->shouldWindowClose())
    {
        // render
        // ------
        window->clear();
        
        Odysseus::SceneGraph::drawScene();

        // HDRImap.update();
        
        window->update();
    }

    delete window;
    return 0;
}
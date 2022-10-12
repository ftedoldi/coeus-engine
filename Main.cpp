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
// #include <CUDAInteroperationManager.hpp>

void runPythonScript()
{
    char filename[] = "test.py";
    FILE* fp;

    Py_Initialize();
    fp = _Py_fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);
    Py_Finalize();
}

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

    // Athena::Quaternion quat = Athena::Quaternion(1, 1, 1, 1);
    // quat.toMatrix4().print();

    //std::thread thread(runPythonScript);

    // Odysseus::Scene* startScene = new Odysseus::Scene(std::string("Start Scene"));
    // Odysseus::SceneManager::addScene(startScene);
    // // TODO: Check if there is a default scene loaded
    // // TODO: Create a default configuration file in order to set the default scene to load
    // Odysseus::SceneManager::setActiveScene(0);


    // //Dummy component test
    // System::Component *c = new DummyComponent();
    // Odysseus::SceneObject *obj = new Odysseus::SceneObject();
    // obj->transform->name = "Object";
    // auto dummy = obj->addComponent<DummyComponent>();

    // Camera setup
    // Odysseus::SceneObject *cam = new Odysseus::SceneObject();
    // cam->transform->position = Athena::Vector3(0, 0, 20);
    // cam->transform->name = "EditorCamera";
    // auto mainEditorCamera = cam->addComponent<Odysseus::EditorCamera>();
    // Odysseus::EditorCamera::main = mainEditorCamera;

    // //Camera movement setup
    // auto movement = cam->addComponent<CameraMovement>();
    // movement->camera = mainCamera;
    // cam->getComponent<CameraMovement>()->camera = cam->getComponent<Odysseus::EditorCamera>();

    // //-------------------------------------------------------
    // //Shaders setup

    Odysseus::Shader* modelShader = new Odysseus::Shader(".\\Shader\\phongShader.vert", ".\\Shader\\phongShader.frag");
    Odysseus::Shader* PBRshader = new Odysseus::Shader(".\\Shader\\PBRshader.vert", ".\\Shader\\PBRshader.frag");

    // //-------------------------------------------------------
    // //Models setup

    // Odysseus::Model myModel("Assets/Models/PBRsphere/PBRsphere.obj", modelShader, false);
    // Odysseus::Model myModel1("Assets/Models/matAndTex/matAndTex.obj", modelShader, false);
    // Odysseus::Model myModel1("Assets/Models/testFiles/posCubesColored.gltf", modelShader, false);
    // // myModel.setIfPBR(true);
    // // myModel1.setIfPBR(true);

    // //-------------------------------------------------------
    // //Light setup

    // //Point light

    // Odysseus::SceneObject *light = new Odysseus::SceneObject();
    // light->transform->name = "PointLight";
    // auto pLight = light->addComponent<Odysseus::PointLight>();
    
    // pLight->setPosition(Athena::Vector3(0.0f, 2.0f, 0.0f));
    // pLight->setShader(modelShader);
    // pLight->setAmbient(Athena::Vector3(0.2f, 0.2f, 0.2f));
    // pLight->setDiffuse(Athena::Vector3(0.8f, 0.8f, 0.8f));
    // pLight->setSpecular(Athena::Vector3(0.3f, 0.3f, 0.3f));
    // pLight->setConstant(1.0f);
    // pLight->setLinear(0.09f);
    // pLight->setQuadratic(0.032f);


    // //Directional light
    // Odysseus::SceneObject *dirLight = new Odysseus::SceneObject();
    // dirLight->transform->name = "DirectionaLight";
    // auto dLight = dirLight->addComponent<Odysseus::DirectionalLight>();

    // dLight->setShader(modelShader);
    // dLight->setAmbient(Athena::Vector3(0.2f, 0.2f, 0.2f));
    // dLight->setDiffuse(Athena::Vector3(0.5f, 0.5f, 0.5f));
    // dLight->setSpecular(Athena::Vector3(0.5f, 0.5f, 0.5f));
    // dLight->setDirection(Athena::Vector3(0.0f, -1.0f, 0.0f));
    
    // //Spot light
    // Odysseus::SceneObject *spotLight = new Odysseus::SceneObject();
    // spotLight->transform->name = "spotLight";
    // auto sLight = spotLight->addComponent<Odysseus::SpotLight>();
    
    // sLight->setPosition(Athena::Vector3(0.0f, 2.0f, 0.0f));
    // sLight->setDirection(Athena::Vector3(0.0f, -1.0f, 0.0f));
    // sLight->setShader(modelShader);
    // sLight->setAmbient(Athena::Vector3(0.2f, 0.2f, 0.2f));
    // sLight->setDiffuse(Athena::Vector3(0.8f, 0.8f, 0.8f));
    // sLight->setSpecular(Athena::Vector3(1.0f, 1.0f, 1.0f));
    // sLight->setCutOff(41.0f);
    // sLight->setSpotExponent(19.0f);

    // Odysseus::SceneObject* obj = new Odysseus::SceneObject();
    // TODO: Do this with rttr
    // std::unique_ptr<System::Component> c = factory::Registry<System::Component>::New("DummyComponent");
    // c->deserialize();
    // Odysseus::SceneObject* obj = new Odysseus::SceneObject();
    // obj->addComponent<DummyComponent>();

    // std::cout << typeid(System::SerializableClass::registeredClasses["DummyComponent"]).name() << std::endl;

    stbi_set_flip_vertically_on_load(true);
    
    // HDR map setup
    Odysseus::Cubemap* HDRImap = new Odysseus::Cubemap();
    HDRImap->setPBRshader(PBRshader);

    // //Setup everything before initializeScene call
    Odysseus::SceneManager::initializeActiveScene();

    // Do this after framebuffer creation
    // CUDA::Interop::CUDARegistration* cudaInterop = new CUDA::Interop::CUDARegistration();
    // cudaInterop->createCUDATextureResource(cudaInterop->cudaResources[0], GL_COLOR_ATTACHMENT0, cudaGraphicsMapFlagsReadOnly);
    // cudaInterop->createCUDATextureResource(cudaInterop->cudaResources[1], 0, cudaGraphicsMapFlagsWriteDiscard);

    // render loop
    // -----------
    while (!window->shouldWindowClose())
    {
        // render
        // ------
        window->clear();
        
        Odysseus::SceneManager::drawActiveScene();

        HDRImap->update();
        
        window->update();
    }

    //thread.join();

    // serializer.serialize(Odysseus::SceneManager::activeScene->path);

    delete window;
    return 0;
}
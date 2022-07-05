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

#include <SceneObject.hpp>
#include <SceneGraph.hpp>
#include <Component.hpp>
#include <Time.hpp>
#include "Test/DummyComponent.hpp"

#include <iostream>
#include <vector>

#include "Scripts/CameraMovement.hpp"

//#include <Test.cuh>

int main()
{
    // Where all the starts are runned
    System::Window* window = new System::Window();

    System::Component *c = new DummyComponent();
    Odysseus::SceneObject *obj = new Odysseus::SceneObject();

    Odysseus::SceneObject *cam = new Odysseus::SceneObject();
    auto mainCamera = cam->addComponent<Odysseus::Camera>();
    auto movement = cam->addComponent<CameraMovement>();
    movement->camera = mainCamera;
    Odysseus::SceneObject *myModel = new Odysseus::SceneObject();
    myModel->addComponent<Odysseus::Model>();
    cam->addComponent<Odysseus::Camera>();

    auto comp = obj->addComponent<DummyComponent>();
    auto comp1 = obj->addComponent<DummyComponent>();
    comp->var = 245;
    comp1->var = 126;
    auto comps = obj->getComponents<DummyComponent>();
    for (int i = 0; i < comps.size(); i++)
        std::cout << comps[i]->var << std::endl;
    std::cout << obj->getComponent<DummyComponent>()->var << std::endl;
    std::cout << obj->removeComponent<DummyComponent>() << std::endl;
    std::cout << obj->getComponent<DummyComponent>()->var << std::endl;
    for (int i = 0; i < comps.size(); i++)
        std::cout << comps[i]->var << std::endl;
    std::cout << obj->getComponents<DummyComponent>().size() << std::endl;

    stbi_set_flip_vertically_on_load(true);

    // Create the shader
    Odysseus::Shader modelShader("shader1.vert", "shader1.frag");

    auto model = myModel->getComponent<Odysseus::Model>();
    model->setPath("Assets/Models/matAndTex/matAndTex.obj");
    model->setShader(&modelShader);
    model->transform->translate(Athena::Vector3(0.0f, 0.0f, -5.0f));

    Odysseus::SceneGraph::initializeScene();

    // render loop
    // -----------
    while (!window->shouldWindowClose())
    {
        // per-frame time logic
        // --------------------
        std::cout << " " << System::Time::time;
        std::cout << " " << System::Time::deltaTime << std::endl;

        // render
        // ------
        window->clear();

        // be sure to activate shader when setting uniforms/drawing objects
        modelShader.use();

        auto camera = cam->getComponent<Odysseus::Camera>();
        model->setCamera(camera);
        // Camera trasformations
        Athena::Matrix4 projection = Odysseus::Camera::perspective(45.0f, System::Window::screen.width / System::Window::screen.height, 0.1f, 100.0f);
        Athena::Matrix4 view = Odysseus::Camera::lookAt(camera->transform->position, camera->transform->position + camera->Front, camera->Up);

        Odysseus::Transform* t = new Odysseus::Transform(Athena::Vector3(0, 0, -3.5), Athena::Quaternion::EulerAnglesToQuaternion(Athena::Vector3(0, 0, 0)), Athena::Vector3(.5, .4, .5));
        t->inverse()->position.print();
        auto tmp = camera->getViewTransform(t);

        modelShader.setMat4("projection", projection);
        //Athena::Matrix4 view = Odysseus::Camera::lookAt(camera->transform->position, camera->transform->position + camera->Front, camera->Up);

        // Model draw
        //myModel.Draw(&modelShader);

        Odysseus::SceneGraph::drawScene();

        window->update();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    delete window;
    return 0;
}
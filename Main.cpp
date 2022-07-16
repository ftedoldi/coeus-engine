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
    // auto movement = cam->addComponent<CameraMovement>();
    // movement->camera = mainCamera;
    // cam->addComponent<CameraMovement>()->camera = cam->getComponent<Odysseus::Camera>();
    // cam->addComponent<Odysseus::Camera>();

    stbi_set_flip_vertically_on_load(true);

    // Create the shader
    Odysseus::Shader modelShader(".\\Shader\\shader1.vert", ".\\Shader\\shader1.frag");

    Odysseus::Model myModel("Assets/Models/cube/untitled.obj", &modelShader);

    Odysseus::SceneGraph::initializeScene();

    // render loop
    // -----------
    while (!window->shouldWindowClose())
    {
        // render
        // ------
        window->clear();

        // be sure to activate shader when setting uniforms/drawing objects
        modelShader.use();

        // auto tmp = cam->getComponent<Odysseus::Camera>()->getViewTransform(new Odysseus::Transform(Athena::Vector3(0, 0, -3.5), Athena::Quaternion(), Athena::Vector3(1, 1, 1)));

        // modelShader.setVec3("position", tmp->position);
        // modelShader.setVec4("rotation", tmp->rotation.asVector4());
        // modelShader.setVec3("scale", tmp->localScale);

        // // Camera trasformations
        // Athena::Matrix4 projection = Odysseus::Camera::perspective(45.0f, System::Window::screen.width / System::Window::screen.height, 0.1f, 100.0f);

        // modelShader.setMat4("projection", projection);

        Odysseus::SceneGraph::drawScene();

        window->update();
    }

    delete window;
    return 0;
}
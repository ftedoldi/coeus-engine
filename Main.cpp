#include <glad/glad.h>
#include <GLFW/glfw3.h>
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
#include "Test/DummyComponent.hpp"

#include <iostream>
#include <vector>

//#include <Test.cuh>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// Utilities variables to mouse callback
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main()
{
    System::Component *c = new DummyComponent();
    Odysseus::SceneObject *obj = new Odysseus::SceneObject();

    // std:: cout << dynamic_cast<DummyComponent*>(c)->var; // WORKS

    Odysseus::SceneObject *cam = new Odysseus::SceneObject();
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
    // obj->container->addComponent(*c);

    // obj->container->components[0]->start();
    // obj->container->components[0]->update();

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state

    stbi_set_flip_vertically_on_load(true);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);

    // Create the shader
    Odysseus::Shader modelShader("shader1.vert", "shader1.frag");

    // Create the model
    Odysseus::Model myModel("Assets/Models/backpack/backpack.obj");

    // Where all the starts are runned
    Odysseus::SceneGraph::initializeScene();

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.4f, 0.2f, 0.6f, 0.5f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // be sure to activate shader when setting uniforms/drawing objects
        modelShader.use();
        modelShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);

        auto camera = cam->getComponent<Odysseus::Camera>();

        // Camera trasformations
        Athena::Matrix4 projection = Odysseus::Camera::perspective(45.0f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        Athena::Matrix4 view = Odysseus::Camera::lookAt(camera->transform->position, camera->transform->position + camera->Front, camera->Up);

        Athena::Quaternion q = Athena::Quaternion::EulerAnglesToQuaternion(Athena::Vector3(0, 20, 50));

        auto tmp = camera->getViewTransform(new Odysseus::Transform(Athena::Vector3(0, 0, -3.5), q, Athena::Vector3(.5, .4, .5)));

        modelShader.setMat4("projection", projection);
        modelShader.setVec3("position", tmp->position);
        modelShader.setVec4("rotation", tmp->rotation.asVector4());
        modelShader.setVec3("scale", tmp->localScale);

        // Model draw
        myModel.Draw(modelShader);

        Odysseus::SceneGraph::drawScene();

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    //     camera.ProcessKeyboard(Odysseus::FORWARD, deltaTime);
    // if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    //     camera.ProcessKeyboard(Odysseus::BACKWARD, deltaTime);
    // if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    //     camera.ProcessKeyboard(Odysseus::LEFT, deltaTime);
    // if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    //     camera.ProcessKeyboard(Odysseus::RIGHT, deltaTime);
    // if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    //     camera.ProcessKeyboard(Odysseus::UP, deltaTime);
    // if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    //     camera.ProcessKeyboard(Odysseus::DOWN, deltaTime);
}
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    // camera.ProcessMouseMovement(xoffset, yoffset);
}
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

//#include <Test.cuh>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

//Creating camera
Odysseus::Camera camera(Athena::Vector3(0.0f, 0.0f, 3.0f));

//Utilities variables to mouse callback
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;


int main()
{
    Odysseus::Component* c = new DummyComponent();
    Odysseus::SceneObject* obj = new Odysseus::SceneObject();

    obj->container->addComponent(*c);

    // obj->container->components[0]->start();
    // obj->container->components[0]->update();

    Odysseus::SceneGraph::initializeScene();

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
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
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

    //Enable depth test
    glEnable(GL_DEPTH_TEST);

    //Create the shader
    Odysseus::Shader modelShader("shader1.vert", "shader1.frag");

    //Create the model
    Odysseus::Model myModel("Assets/Models/matAndTex/matAndTex.obj");

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        Odysseus::SceneGraph::drawScene();

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

        //Camera trasformations
		Athena::Matrix4 projection = Odysseus::Camera::perspective(45.0f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        Athena::Matrix4 view = Odysseus::Camera::lookAt(camera.Position, camera.Position + camera.Front, camera.Up);

        modelShader.setMat4("projection", projection);
        modelShader.setMat4("view", view);

        //Model trasformations
		Athena::Matrix4 model;
		//model = Athena::Matrix4::translate(model, Athena::Vector3(0.2f, 0.2f, 0.2f));
		//model = Athena::Matrix4::translate(model, Athena::Vector3(0.4f, 0.4f, 0.2f));
		//model = Athena::Matrix4::scale(model, Athena::Vector3(0.4f, 0.4f, 0.4f));
		model = Athena::Matrix4::scale(model, Athena::Vector3(0.4f, 0.4f, 0.4f));
		modelShader.setMat4("model", model);

        //Model draw
        myModel.Draw(modelShader);

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

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(Odysseus::FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(Odysseus::BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(Odysseus::LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(Odysseus::RIGHT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.ProcessKeyboard(Odysseus::UP, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.ProcessKeyboard(Odysseus::DOWN, deltaTime);
}
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
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

    camera.ProcessMouseMovement(xoffset, yoffset);
}
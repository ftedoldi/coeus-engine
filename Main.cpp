#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <assimp/Importer.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <glm/vec2.hpp>

#include <Vector2.hpp>
#include <Vector3.hpp>
#include <Vector4.hpp>
#include <Quaternion.hpp>

#include <iostream>

//#include <Test.cuh>

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

int main() {
	//call();
	Athena::Quaternion::TestClass();
	Athena::Quaternion quat = Athena::Quaternion::EulerAnglesToQuaternion(Athena::Vector3(1, 2, 3));
	Athena::Quaternion::QuaternionToEulerAngles(quat).print();
	Athena::Vector4(Athena::Vector3(2, 1, 2), 1.0f).print();
	std::cout << Athena::Vector4().dot(Athena::Vector4()) << std::endl;
	Athena::Vector4(Athena::Vector3(2, 1, 2), 1.0f).print();
	std::cout << (Athena::Vector4(1, 2, 3, 4) == Athena::Vector4(1, 2, 3, 4));
	Athena::Vector2 vec = Athena::Vector2(1, 2);
	vec.lerp(Athena::Vector2(), 0.5f).print();

	vec.cross(Athena::Vector2(2, 30)).print();

    glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);

	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	// render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window);

		// render
		// ------
		// glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// glfw: terminate, clearing all previously allocated GLFWresources.
	//---------------------------------------------------------------
	glfwTerminate();
	return 0;
}
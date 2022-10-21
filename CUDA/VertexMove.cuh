#ifndef VERTEXMOVE_H
#define VERTEXMOVE_H

#ifdef __cplusplus
extern "C" {
#endif

void __declspec(dllexport) MoveVertices(cudaGraphicsResource_t& vbo, size_t numVertices);

#ifdef __cplusplus
}
#endif

#endif  // KERNEL_H
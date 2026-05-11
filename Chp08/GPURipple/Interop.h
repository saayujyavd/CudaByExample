#ifndef INTEROP_H
#define INTEROP_H

#include <gl/glew.h>
#include <gl/glut.h>
#include <cuda/cuda.h>
#include <cuda/cuda_gl_interop.h>
#include <book.h>
#include <cpu_bitmap.h>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "freeglut.lib")

#define DIMX	1920
#define DIMY	1080

__global__ void kernel(uchar4* ptr, int);

namespace CudaGLInterop
{
	inline int selectCudaDev(cudaDeviceProp& prop)
	{
		unsigned int cnt;
		int dev;

		HANDLE_ERROR(cudaGLGetDevices(&cnt, &dev, 1, cudaGLDeviceListAll));
		if (cnt == 0)
		{
			printf("No CUDA device supports GL interop\n");
			exit(EXIT_FAILURE);
		}

		return(dev);
	}

	inline void initOgl()
	{
		int c = 1;
		char* foo[] = { (char*)"name" };
		
		glutInit(&c, foo);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(DIMX, DIMY);
		glutCreateWindow("Ripple");
		
		glewInit();
	}

	inline void createOglPbo(GLuint* buffer_obj)
	{
		glGenBuffers(1, buffer_obj);								// gen a buff handle
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *buffer_obj);		// bind the handle to a pixel buffer (PB)
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIMX * DIMY * 4,
			NULL, GL_DYNAMIC_DRAW_ARB);								// request OGL driver to allocate a buff
	}

	inline void setInterop(GLuint* buffer_obj, cudaGraphicsResource** resource)
	{
		// init the OpenGL driver
		initOgl();

		// select a CUDA device on which to run this application
		cudaDeviceProp prop;
		int dev = selectCudaDev(prop);

		// tell the CUDA runtime that intention is to use the device for CUDA and OpenGL
		HANDLE_ERROR(cudaGLSetGLDevice(dev));

		// create a pixel buffer object (PBO) in OpenGL and store handle in 'buffer_obj'
		createOglPbo(buffer_obj);

		// notify CUDA runtime that intention is to share 'buffer_obj' with CUDA
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(resource, *buffer_obj,
			cudaGraphicsMapFlagsNone));	// cudaGraphicsMapFlagsNone: no specific behaviour of this buff
	}

	template<typename T>
	uchar4* getDevicePointer(T* bitmap)
	{
		uchar4* dev_ptr;
		size_t size;

		HANDLE_ERROR(cudaGraphicsMapResources(1, &(bitmap->resource), NULL));
		HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, bitmap->resource));

		return(dev_ptr);
	}

	template<typename T>
	void launchCudaKernel(T* bitmap)
	{
		static int ticks = 1;

		// instruct the CUDA runtime to map the shared rsrc and then request a ptr. to mapped rsrc
		uchar4* dev_ptr = getDevicePointer<T>(bitmap);

		bitmap->fAnim(dev_ptr, bitmap->data_block, ticks++);

		// provide synchronization between CUDA and Graphics portions of application
		//HANDLE_ERROR(cudaDeviceSynchronize());
		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &(bitmap->resource), NULL)); /* <-- very important */

		glutPostRedisplay();
	}

	static void drawFunc()
	{
		glDrawPixels(DIMX, DIMY, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glutSwapBuffers();
	}

	inline void setGlutMainLoop(void (*keyFunc)(unsigned char, int, int),
								void (*idleFunc)(void))
	{
		glutKeyboardFunc(keyFunc);
		glutDisplayFunc(drawFunc);
		glutIdleFunc(idleFunc);
		glutMainLoop();
	}

	inline void endProcess(GLuint buffer_obj, cudaGraphicsResource* resource)
	{
		HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &buffer_obj);

		exit(0);
	}
}

#endif

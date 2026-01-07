#pragma once

#include <cstddef>
#include <cstdint>

/**
 * \file metal_backend.hpp
 * \brief Minimal Metal compute wrapper used by the Apple MPS path.
 */

/**
 * \brief Opaque Metal buffer handle and size in bytes.
 */
struct MetalBuffer
{
	void *handle = nullptr;
	size_t size = 0;
};

/**
 * \brief Buffer binding for a dispatch (buffer + offset).
 */
struct MetalDispatchBuffer
{
	const MetalBuffer *buffer = nullptr;
	size_t offset = 0;
};

/**
 * \brief Inline POD bytes for a dispatch (small constant args).
 */
struct MetalDispatchBytes
{
	const void *bytes = nullptr;
	size_t length = 0;
};

/**
 * \brief Opaque Metal texture handle for kernels that accept textures.
 */
struct MetalDispatchTexture
{
	void *handle = nullptr;
};

#ifdef DARKNET_USE_MPS

/** \defgroup metal_backend Metal Backend
 *  \brief Minimal Metal compute wrapper used by the Apple MPS path.
 *  @{
 */

bool metal_is_available();
bool metal_init();
void metal_shutdown();

/**
 * \brief Begin/end a logical frame (enqueue work for a command buffer).
 */
void metal_begin_frame();
void metal_end_frame();
void metal_flush();

bool metal_buffer_alloc(size_t size, MetalBuffer *out);
void metal_buffer_free(MetalBuffer *buffer);
bool metal_buffer_upload(const MetalBuffer *buffer, size_t offset, const void *data, size_t length);
bool metal_buffer_download(const MetalBuffer *buffer, size_t offset, void *data, size_t length);
bool metal_buffer_fill(const MetalBuffer *buffer, size_t offset, uint8_t value, size_t length);

/**
 * \brief Dispatch a 1D compute kernel.
 */
bool metal_dispatch_1d(const char *kernel_name,
	size_t threads, size_t threads_per_group,
	const MetalDispatchTexture *textures, size_t texture_count,
	const MetalDispatchBuffer *buffers, size_t buffer_count,
	const MetalDispatchBytes *bytes, size_t bytes_count,
	void *command_buffer);

/**
 * \brief Dispatch a 2D compute kernel.
 */
bool metal_dispatch_2d(const char *kernel_name,
	size_t width, size_t height,
	size_t threads_per_group_x, size_t threads_per_group_y,
	const MetalDispatchTexture *textures, size_t texture_count,
	const MetalDispatchBuffer *buffers, size_t buffer_count,
	const MetalDispatchBytes *bytes, size_t bytes_count,
	void *command_buffer);

/**
 * \brief Dispatch a kernel using a texture as the grid shape source.
 */
bool metal_dispatch_texture_kernel(const char *kernel_name,
	void *grid_texture_handle,
	size_t threads_per_group_x, size_t threads_per_group_y,
	const MetalDispatchTexture *textures, size_t texture_count,
	const MetalDispatchBuffer *buffers, size_t buffer_count,
	const MetalDispatchBytes *bytes, size_t bytes_count,
	void *command_buffer);

bool metal_self_test();

/** @} */
#else

static inline bool metal_is_available()
{
	return false;
}

static inline bool metal_init()
{
	return false;
}

static inline void metal_shutdown()
{
	return;
}

static inline void metal_begin_frame()
{
	return;
}

static inline void metal_end_frame()
{
	return;
}

static inline void metal_flush()
{
	return;
}

static inline bool metal_buffer_alloc(size_t, MetalBuffer *)
{
	return false;
}

static inline void metal_buffer_free(MetalBuffer *)
{
	return;
}

static inline bool metal_buffer_upload(const MetalBuffer *, size_t, const void *, size_t)
{
	return false;
}

static inline bool metal_buffer_download(const MetalBuffer *, size_t, void *, size_t)
{
	return false;
}

static inline bool metal_buffer_fill(const MetalBuffer *, size_t, uint8_t, size_t)
{
	return false;
}

static inline bool metal_dispatch_1d(const char *, size_t, size_t, const MetalDispatchTexture *, size_t, const MetalDispatchBuffer *, size_t, const MetalDispatchBytes *, size_t, void *)
{
	return false;
}

static inline bool metal_dispatch_2d(const char *, size_t, size_t, size_t, size_t, const MetalDispatchTexture *, size_t, const MetalDispatchBuffer *, size_t, const MetalDispatchBytes *, size_t, void *)
{
	return false;
}

static inline bool metal_dispatch_texture_kernel(const char *, void *, size_t, size_t, const MetalDispatchTexture *, size_t, const MetalDispatchBuffer *, size_t, const MetalDispatchBytes *, size_t, void *)
{
	return false;
}

static inline bool metal_self_test()
{
	return false;
}

#endif

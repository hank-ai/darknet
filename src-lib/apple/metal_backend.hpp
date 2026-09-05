#pragma once

/** @file
 * Apple backend Metal compute facade.
 */

#include <cstddef>
#include <cstdint>
#include <string>

#ifndef DARKNET_USE_MPS
#error "metal_backend.hpp is private to builds with Apple MPS enabled"
#endif

namespace Darknet::Apple
{

	/** @defgroup metal_backend Metal Backend
 * Metal compute wrapper used by the Apple backend.
 * @{
 */

	/** Non-owning view of an opaque Metal buffer. */
	struct MetalBufferView
	{
		void *handle = nullptr; ///< Objective-C Metal buffer handle.
		size_t size = 0;		///< Buffer size in bytes.
	};

	/** Backward-compatible name used by the procedural allocation API. */
	using MetalBuffer = MetalBufferView;

	/** Buffer binding for a dispatch. */
	struct MetalDispatchBuffer
	{
		const MetalBufferView *buffer = nullptr; ///< Buffer to bind.
		size_t offset = 0;						 ///< Offset into @ref buffer, in bytes.
	};

	/** Inline POD bytes for a dispatch, normally used for small constant arguments. */
	struct MetalDispatchBytes
	{
		const void *bytes = nullptr; ///< Raw bytes to bind.
		size_t length = 0;			 ///< Number of bytes to bind.
	};

	/** Opaque Metal texture handle for kernels that accept textures. */
	struct MetalDispatchTexture
	{
		void *handle = nullptr; ///< Objective-C Metal texture handle.
	};

	/// Return @p true if a Metal device and command queue can be created.
	bool metal_is_available();

	/// Initialize the shared Metal runtime.
	bool metal_init();

	/// Begin a logical frame so multiple dispatches can share a command buffer.
	void metal_begin_frame();

	/// Commit and wait for the current logical frame.
	void metal_end_frame();

	/// Commit and wait for any outstanding thread-local Metal work.
	void metal_flush();

	/// Allocate a Metal buffer.
	bool metal_buffer_alloc(size_t size, MetalBuffer *out);

	/// Release a Metal buffer and reset its descriptor.
	void metal_buffer_free(MetalBuffer *buffer);

	/// Copy host memory into a Metal buffer.
	bool metal_buffer_upload(const MetalBuffer *buffer, size_t offset, const void *data, size_t length);

	/// Copy Metal buffer memory back to the host.
	bool metal_buffer_download(const MetalBuffer *buffer, size_t offset, void *data, size_t length);

	/// Fill a Metal buffer range with @p value.
	bool metal_buffer_fill(const MetalBuffer *buffer, size_t offset, uint8_t value, size_t length);

	/// Return @p true if a Metal compute kernel can be compiled.
	bool metal_kernel_available(const std::string & kernel_name);

	/// Return the most recent Metal compilation, pipeline, or submission error.
	std::string metal_last_error();

	/// Dispatch a 1D compute kernel.
	bool metal_dispatch_1d(const std::string & kernel_name,
		size_t threads, size_t threads_per_group,
		const MetalDispatchTexture *textures, size_t texture_count,
		const MetalDispatchBuffer *buffers, size_t buffer_count,
		const MetalDispatchBytes *bytes, size_t bytes_count,
		void *command_buffer);

	/// Dispatch a 1D compute kernel using an existing compute encoder.
	bool metal_dispatch_1d_with_encoder(const std::string & kernel_name,
		size_t threads, size_t threads_per_group,
		const MetalDispatchTexture *textures, size_t texture_count,
		const MetalDispatchBuffer *buffers, size_t buffer_count,
		const MetalDispatchBytes *bytes, size_t bytes_count,
		void *encoder_handle);

	/// Dispatch a 2D compute kernel.
	bool metal_dispatch_2d(const std::string & kernel_name,
		size_t width, size_t height,
		size_t threads_per_group_x, size_t threads_per_group_y,
		const MetalDispatchTexture *textures, size_t texture_count,
		const MetalDispatchBuffer *buffers, size_t buffer_count,
		const MetalDispatchBytes *bytes, size_t bytes_count,
		void *command_buffer);

	/// Dispatch a kernel using a texture as the grid shape source.
	bool metal_dispatch_texture_kernel(const std::string & kernel_name,
		void *grid_texture_handle,
		size_t threads_per_group_x, size_t threads_per_group_y,
		const MetalDispatchTexture *textures, size_t texture_count,
		const MetalDispatchBuffer *buffers, size_t buffer_count,
		const MetalDispatchBytes *bytes, size_t bytes_count,
		void *command_buffer);

	/** Move-only owner for buffers allocated by @ref metal_buffer_alloc(). */
	class MetalBufferOwner
	{
		public:
		MetalBufferOwner() = default;
		MetalBufferOwner(const MetalBufferOwner &) = delete;
		MetalBufferOwner & operator=(const MetalBufferOwner &) = delete;

		MetalBufferOwner(MetalBufferOwner && other) noexcept : buffer_(other.release())
		{
		}

		MetalBufferOwner & operator=(MetalBufferOwner && other) noexcept
		{
			if (this != &other)
			{
				reset();
				buffer_ = other.release();
			}
			return *this;
		}

		~MetalBufferOwner()
		{
			reset();
		}

		bool allocate(size_t size)
		{
			reset();
			return metal_buffer_alloc(size, &buffer_);
		}

		void reset()
		{
			metal_buffer_free(&buffer_);
		}

		MetalBufferView *get()
		{
			return &buffer_;
		}

		const MetalBufferView *get() const
		{
			return &buffer_;
		}

		bool upload(size_t offset, const void *data, size_t length) const
		{
			return metal_buffer_upload(&buffer_, offset, data, length);
		}

		bool download(size_t offset, void *data, size_t length) const
		{
			return metal_buffer_download(&buffer_, offset, data, length);
		}

		explicit operator bool() const
		{
			return buffer_.handle != nullptr;
		}

	private:
		MetalBufferView release()
		{
			const MetalBufferView buffer = buffer_;
			buffer_ = {};
			return buffer;
		}

		MetalBufferView buffer_;
	};

	/** @} */

} // namespace Darknet::Apple
